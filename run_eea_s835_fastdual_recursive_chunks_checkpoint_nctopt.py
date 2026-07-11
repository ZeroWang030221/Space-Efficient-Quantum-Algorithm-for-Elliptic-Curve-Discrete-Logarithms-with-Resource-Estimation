import argparse
import gc
import json
import multiprocessing as _mp
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import eea_circuit_s835_fastdual as eea
from ccx_recursive_block_counter import CounterPolicy
from nct_template_segment_optimizer import NCTTemplatePolicy, count_gate_or_circuit_nct_optimized


def _json_counter(c: Counter) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(c.items())}


def _counter(d: dict[str, Any] | None) -> Counter:
    return Counter({str(k): int(v) for k, v in (d or {}).items() if isinstance(v, int) or (isinstance(v, float) and not isinstance(v, bool))})


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _policy_matches(data: dict[str, Any], *, tpolicy: NCTTemplatePolicy, cpolicy: CounterPolicy) -> bool:
    # Fallback steps store the requested policy in nct_template_policy and the
    # actual rounds=0 policy in actual_nct_template_policy.  They should be
    # resumable under the requested policy because their counts are complete.
    return data.get("nct_template_policy") == tpolicy.as_dict() and data.get("counter_policy") == cpolicy.as_dict()


def _meta_from_ops(ops: Counter) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(ops.items()) if str(k).startswith("meta::")}


def _policy_with(template_policy: NCTTemplatePolicy, **updates: Any) -> NCTTemplatePolicy:
    d = template_policy.as_dict()
    d.update(updates)
    return NCTTemplatePolicy(**d)


def step_path_for(workdir: Path, n: int, T: int) -> Path:
    return workdir / "steps" / f"eea_s835_fastdual_n{n}_T{T:04d}_nctopt_step.json"


def chunk_path_for(workdir: Path, n: int, start: int, end: int) -> Path:
    return workdir / f"eea_s835_fastdual_n{n}_T{start:04d}_{end:04d}_nctopt.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _count_step_direct(
    n: int,
    T_max: int,
    T: int,
    *,
    aux_size: int,
    measurement_uncompute: bool,
    tpolicy: NCTTemplatePolicy,
    cpolicy: CounterPolicy,
    reported_tpolicy: NCTTemplatePolicy | None = None,
    fallback_reason: str | None = None,
    trace_step: bool = False,
) -> dict[str, Any]:
    cfg = eea.get_n_config(n)
    lw = int(cfg["len_width"])
    sw = int(cfg["shift_width"])
    eea.set_measurement_uncompute(measurement_uncompute)

    tic = time.perf_counter()
    if trace_step:
        print(f"    [build-step] T={T}/{T_max} start", flush=True)
    qc = eea.build_step_circuit(n, T, T_max=T_max, aux_size=aux_size, measurement_uncompute=measurement_uncompute)
    if trace_step:
        print(f"    [built-step] T={T}/{T_max} qubits={qc.num_qubits} top_ops={len(qc.data)} count start", flush=True)
    ops = count_gate_or_circuit_nct_optimized(qc, counter_policy=cpolicy, template_policy=tpolicy)
    if fallback_reason:
        ops["meta::nct_step_timeout_fallback_unoptimized"] += 1
    elapsed = time.perf_counter() - tic
    num_qubits = int(qc.num_qubits)
    del qc
    if trace_step:
        print(f"    [counted-step] T={T}/{T_max} elapsed={elapsed:.2f}s ccx={ops.get('ccx', 0)} cx={ops.get('cx', 0)}", flush=True)

    reported = reported_tpolicy or tpolicy
    out = {
        "mode": "eea-s835-fastdual-recursive-step-nctopt-bounded-failopen",
        "n": int(n),
        "T_max": int(T_max),
        "T": int(T),
        "range": [int(T), int(T)],
        "num_qubits": num_qubits,
        "len_width": lw,
        "shift_width": sw,
        "aux_size": int(aux_size),
        "measurement_based": bool(measurement_uncompute),
        "nct_optimized": True,
        "fail_open_unchanged_on_timeout_or_exception": True,
        "nct_template_policy": reported.as_dict(),
        "actual_nct_template_policy": tpolicy.as_dict(),
        "counter_policy": cpolicy.as_dict(),
        "ops": _json_counter(ops),
        "key_counts": {
            "toffoli": int(ops.get("ccx", 0)),
            "ccx": int(ops.get("ccx", 0)),
            "cx": int(ops.get("cx", 0)),
        },
        "optimization_meta": _meta_from_ops(ops),
        "elapsed_s": float(elapsed),
    }
    if fallback_reason:
        out["fallback_reason"] = str(fallback_reason)
    return out


def _best_mp_context() -> Any:
    try:
        methods = _mp.get_all_start_methods()
        if "fork" in methods:
            return _mp.get_context("fork")
    except Exception:
        pass
    return _mp.get_context("spawn")


def _count_step_worker(
    conn: Any,
    n: int,
    T_max: int,
    T: int,
    aux_size: int,
    measurement_uncompute: bool,
    tpolicy: NCTTemplatePolicy,
    cpolicy: CounterPolicy,
    reported_tpolicy: NCTTemplatePolicy | None,
    fallback_reason: str | None,
    trace_step: bool,
) -> None:
    try:
        data = _count_step_direct(
            n,
            T_max,
            T,
            aux_size=aux_size,
            measurement_uncompute=measurement_uncompute,
            tpolicy=tpolicy,
            cpolicy=cpolicy,
            reported_tpolicy=reported_tpolicy,
            fallback_reason=fallback_reason,
            trace_step=trace_step,
        )
        conn.send(("ok", data, None))
    except Exception:
        try:
            conn.send(("exception", None, traceback.format_exc()))
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _run_step_in_subprocess(
    *,
    n: int,
    T_max: int,
    T: int,
    aux_size: int,
    measurement_uncompute: bool,
    tpolicy: NCTTemplatePolicy,
    cpolicy: CounterPolicy,
    timeout_s: float,
    reported_tpolicy: NCTTemplatePolicy | None = None,
    fallback_reason: str | None = None,
    trace_step: bool = False,
) -> tuple[str, dict[str, Any] | None, str | None]:
    if float(timeout_s) <= 0:
        try:
            data = _count_step_direct(
                n,
                T_max,
                T,
                aux_size=aux_size,
                measurement_uncompute=measurement_uncompute,
                tpolicy=tpolicy,
                cpolicy=cpolicy,
                reported_tpolicy=reported_tpolicy,
                fallback_reason=fallback_reason,
                trace_step=trace_step,
            )
            return "ok", data, None
        except Exception:
            return "exception", None, traceback.format_exc()

    parent = child = proc = None
    try:
        ctx = _best_mp_context()
        parent, child = ctx.Pipe(duplex=False)
        proc = ctx.Process(
            target=_count_step_worker,
            args=(child, n, T_max, T, aux_size, measurement_uncompute, tpolicy, cpolicy, reported_tpolicy, fallback_reason, trace_step),
        )
        # Do not use daemon=True; if timeout_mode=process is requested inside the
        # worker, it must be allowed to create its own child for local windows.
        proc.daemon = False
        proc.start()
        child.close()
        child = None

        if parent.poll(float(timeout_s)):
            status, data, err = parent.recv()
            proc.join(timeout=2.0)
            if status == "ok":
                return "ok", data, None
            return "exception", None, str(err)

        proc.terminate()
        proc.join(timeout=2.0)
        if proc.is_alive():
            try:
                proc.kill()
            except Exception:
                pass
            proc.join(timeout=2.0)
        return "timeout", None, None
    except Exception:
        return "exception", None, traceback.format_exc()
    finally:
        for obj in (parent, child):
            try:
                if obj is not None:
                    obj.close()
            except Exception:
                pass


def count_step_failopen(
    n: int,
    T_max: int,
    T: int,
    *,
    aux_size: int,
    measurement_uncompute: bool,
    tpolicy: NCTTemplatePolicy,
    cpolicy: CounterPolicy,
    step_timeout_s: float,
    fallback_step_timeout_s: float,
    trace_step: bool = False,
) -> dict[str, Any]:
    """Count one step; on optimized timeout/exception, recount unchanged."""
    status, data, err = _run_step_in_subprocess(
        n=n,
        T_max=T_max,
        T=T,
        aux_size=aux_size,
        measurement_uncompute=measurement_uncompute,
        tpolicy=tpolicy,
        cpolicy=cpolicy,
        timeout_s=float(step_timeout_s),
        trace_step=trace_step,
    )
    if status == "ok" and data is not None:
        return data

    reason = f"optimized_step_{status}"
    detail = f" after {float(step_timeout_s):.1f}s" if status == "timeout" and float(step_timeout_s) > 0 else ""
    print(
        f"  [step-fallback] T={T}/{T_max} {reason}{detail}; recounting this step unchanged with rounds=0",
        flush=True,
    )
    if err:
        print(f"  [step-fallback-detail] T={T}: {err.splitlines()[-1] if err.splitlines() else err}", flush=True)

    fallback_policy = _policy_with(
        tpolicy,
        rounds=0,
        segment_timeout_s=0.0,
        timeout_mode="none",
        optimization_budget_s=0.0,
        max_template_attempts_per_count=0,
        fast_cancel=False,
        progress=False,
    )
    fb_status, fb_data, fb_err = _run_step_in_subprocess(
        n=n,
        T_max=T_max,
        T=T,
        aux_size=aux_size,
        measurement_uncompute=measurement_uncompute,
        tpolicy=fallback_policy,
        cpolicy=cpolicy,
        timeout_s=float(fallback_step_timeout_s),
        reported_tpolicy=tpolicy,
        fallback_reason=f"{reason}{detail}",
        trace_step=trace_step,
    )
    if fb_status == "ok" and fb_data is not None:
        return fb_data
    if fb_status == "timeout":
        raise RuntimeError(
            f"fallback unchanged count for T={T} also timed out after {float(fallback_step_timeout_s):.1f}s; "
            "cannot safely produce complete counts for this step"
        )
    raise RuntimeError(f"fallback unchanged count failed for T={T}:\n{fb_err}")


def count_range(
    n: int,
    T_max: int,
    start: int,
    end: int,
    *,
    aux_size: int,
    measurement_uncompute: bool,
    tpolicy: NCTTemplatePolicy,
    cpolicy: CounterPolicy,
    workdir: Path | None = None,
    resume: bool = False,
    ignore_policy_mismatch: bool = False,
    force: bool = False,
    print_every: int = 1,
    step_timeout_s: float = 60.0,
    fallback_step_timeout_s: float = 900.0,
    trace_step: bool = False,
) -> dict[str, Any]:
    cfg = eea.get_n_config(n)
    lw = int(cfg["len_width"])
    sw = int(cfg["shift_width"])
    total: Counter = Counter()
    num_qubits: int | None = None
    step_summaries: list[dict[str, Any]] = []

    for T in range(start, end + 1):
        spath = step_path_for(workdir, n, T) if workdir is not None else None
        data: dict[str, Any] | None = None
        if resume and not force and spath is not None and spath.exists() and spath.stat().st_size > 0:
            candidate = load_json(spath)
            if ignore_policy_mismatch or _policy_matches(candidate, tpolicy=tpolicy, cpolicy=cpolicy):
                data = candidate
                if print_every > 0:
                    print(f"  [resume-step] T={T}/{T_max} ccx={data['ops'].get('ccx', 0)} cx={data['ops'].get('cx', 0)}", flush=True)
            else:
                print(f"  [stale-step] T={T}: policy mismatch, recounting", flush=True)

        if data is None:
            if print_every > 0 and ((T - start) % print_every == 0):
                print(f"  [count-step] T={T}/{T_max} start", flush=True)
            data = count_step_failopen(
                n,
                T_max,
                T,
                aux_size=aux_size,
                measurement_uncompute=measurement_uncompute,
                tpolicy=tpolicy,
                cpolicy=cpolicy,
                step_timeout_s=step_timeout_s,
                fallback_step_timeout_s=fallback_step_timeout_s,
                trace_step=trace_step,
            )
            if spath is not None:
                _atomic_write_json(spath, data)
            meta = data.get("optimization_meta", {})
            print(
                f"  [done-step] T={T}/{T_max} {data.get('elapsed_s', 0):.2f}s "
                f"ccx={data['ops'].get('ccx', 0)} cx={data['ops'].get('cx', 0)} "
                f"qiskit_ok={meta.get('meta::nct_segments_qiskit_optimized', 0)} "
                f"changed={meta.get('meta::nct_segments_changed', 0)} "
                f"skip_q={meta.get('meta::nct_segments_skipped_many_qubits', 0)} "
                f"timeout={meta.get('meta::nct_segments_skipped_timeout', 0)} "
                f"step_fallback={meta.get('meta::nct_step_timeout_fallback_unoptimized', 0)}",
                flush=True,
            )

        ops = _counter(data.get("ops", {}))
        total += ops
        num_qubits = int(data.get("num_qubits", num_qubits or 0))
        step_summaries.append({
            "T": int(T),
            "range": [int(T), int(T)],
            "ops": _json_counter(ops),
            "key_counts": {
                "toffoli": int(ops.get("ccx", 0)),
                "ccx": int(ops.get("ccx", 0)),
                "cx": int(ops.get("cx", 0)),
            },
            "elapsed_s": float(data.get("elapsed_s", 0.0)),
            "optimization_meta": data.get("optimization_meta", {}),
            "fallback_reason": data.get("fallback_reason"),
            "actual_nct_template_policy": data.get("actual_nct_template_policy"),
        })
        if T % 25 == 0:
            gc.collect()

    return {
        "mode": "eea-s835-fastdual-recursive-chunk-nctopt-bounded-failopen",
        "n": int(n),
        "T_max": int(T_max),
        "range": [int(start), int(end)],
        "num_qubits": int(num_qubits or 0),
        "len_width": lw,
        "shift_width": sw,
        "aux_size": int(aux_size),
        "measurement_based": bool(measurement_uncompute),
        "nct_optimized": True,
        "fail_open_unchanged_on_timeout_or_exception": True,
        "whole_step_timeout_fallback_enabled": bool(float(step_timeout_s) > 0),
        "nct_template_policy": tpolicy.as_dict(),
        "counter_policy": cpolicy.as_dict(),
        "ops": _json_counter(total),
        "key_counts": {
            "toffoli": int(total.get("ccx", 0)),
            "ccx": int(total.get("ccx", 0)),
            "cx": int(total.get("cx", 0)),
        },
        "optimization_meta": _meta_from_ops(total),
        "steps": step_summaries,
    }


def write_sum(
    out_path: Path,
    *,
    n: int,
    T_max: int,
    chunks: list[dict[str, Any]],
    elapsed_s: float,
    tpolicy: NCTTemplatePolicy,
    cpolicy: CounterPolicy,
) -> None:
    total: Counter = Counter()
    q = lw = sw = aux = None
    for c in chunks:
        total += _counter(c.get("ops", {}))
        q = int(c["num_qubits"])
        lw = int(c["len_width"])
        sw = int(c["shift_width"])
        aux = int(c["aux_size"])
    out = {
        "mode": "eea-s835-fastdual-recursive-chunks-checkpointed-nctopt-bounded-failopen",
        "n": int(n),
        "T_max": int(T_max),
        "range": [1, int(T_max)] if chunks and chunks[0]["range"][0] == 1 and chunks[-1]["range"][1] == T_max else None,
        "num_qubits": q,
        "len_width": lw,
        "shift_width": sw,
        "aux_size": aux,
        "measurement_based": bool(chunks[0].get("measurement_based", False)) if chunks else None,
        "nct_optimized": True,
        "fail_open_unchanged_on_timeout_or_exception": True,
        "nct_template_policy": tpolicy.as_dict(),
        "counter_policy": cpolicy.as_dict(),
        "ops": _json_counter(total),
        "key_counts": {
            "toffoli": int(total.get("ccx", 0)),
            "ccx": int(total.get("ccx", 0)),
            "cx": int(total.get("cx", 0)),
        },
        "optimization_meta": _meta_from_ops(total),
        "chunks": [
            {
                "range": c["range"],
                "ops": {str(k): int(v) for k, v in c["ops"].items()},
                "key_counts": c.get("key_counts", {}),
                "optimization_meta": c.get("optimization_meta", {}),
            }
            for c in chunks
        ],
        "elapsed_s_so_far": float(elapsed_s),
    }
    _atomic_write_json(out_path, out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Bounded fail-open checkpointed NCT-template optimized EEA Algorithm-3 counter")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--T-max", type=int, default=None)
    ap.add_argument("--chunk-size", type=int, default=25)
    ap.add_argument("--aux-size", type=int, default=None)
    ap.add_argument("--workdir", default="eea_s835_fastdual_chunks_nctopt_bounded_failopen")
    ap.add_argument("--out", default="eea_s835_fastdual_algorithm3_recursive_chunks_n256_measurement_nctopt_bounded_failopen.json")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force", action="store_true", help="Recompute even if step/chunk checkpoints exist")
    ap.add_argument("--ignore-policy-mismatch", action="store_true", help="Reuse old checkpoints even if NCT policy differs; normally do not use this")
    ap.add_argument("--measurement-uncompute", action="store_true")
    ap.add_argument("--templates", choices=["small-nct", "all-nct"], default="small-nct")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--max-nct-segment-gates", type=int, default=12)
    ap.add_argument("--max-nct-segment-qubits", type=int, default=6)
    ap.add_argument("--min-nct-segment-gates", type=int, default=2)
    ap.add_argument("--segment-timeout-s", type=float, default=1.0)
    ap.add_argument("--timeout-mode", choices=["auto", "signal", "process", "none"], default="auto")
    ap.add_argument("--optimization-budget-s", type=float, default=20.0)
    ap.add_argument("--max-template-attempts-per-count", "--max-template-attempts", dest="max_template_attempts_per_count", type=int, default=80)
    ap.add_argument("--keep-trying-after-timeout", action="store_true", help="Do not disable template attempts after the first local timeout")
    ap.add_argument("--disable-fast-cancel", action="store_true", help="Disable deterministic adjacent self-inverse NCT cancellation")
    ap.add_argument("--step-timeout-s", type=float, default=60.0, help="Whole optimized T-step timeout; <=0 disables. On timeout/exception the step is recounted exactly with rounds=0.")
    ap.add_argument("--fallback-step-timeout-s", type=float, default=900.0, help="Timeout for exact unchanged fallback step count; 0 means wait indefinitely.")
    ap.add_argument("--trace-step", action="store_true", help="Print build/count milestones inside each T step")
    ap.add_argument("--heuristics-backward-length", type=int, default=3)
    ap.add_argument("--heuristics-backward-survivor", type=int, default=1)
    ap.add_argument("--heuristics-qubits-length", type=int, default=1)
    ap.add_argument("--ccx-cost", type=int, default=100)
    ap.add_argument("--cx-cost", type=int, default=1)
    ap.add_argument("--x-cost", type=int, default=1)
    ap.add_argument("--print-every", type=int, default=1)
    args = ap.parse_args()

    cfg = eea.get_n_config(args.n)
    T_max = int(args.T_max or cfg["T_max"])
    lw = int(cfg["len_width"])
    sw = int(cfg["shift_width"])
    if args.aux_size is None:
        args.aux_size = int(eea.qiskit_paper_aux_size(args.n, lw, sw, T_max))

    tpolicy = NCTTemplatePolicy(
        templates=args.templates,
        rounds=args.rounds,
        max_segment_gates=args.max_nct_segment_gates,
        max_segment_qubits=args.max_nct_segment_qubits,
        min_segment_gates=args.min_nct_segment_gates,
        segment_timeout_s=args.segment_timeout_s,
        timeout_mode=args.timeout_mode,
        optimization_budget_s=args.optimization_budget_s,
        max_template_attempts_per_count=args.max_template_attempts_per_count,
        disable_templates_after_timeout=not args.keep_trying_after_timeout,
        fast_cancel=not args.disable_fast_cancel,
        heuristics_backward_length=args.heuristics_backward_length,
        heuristics_backward_survivor=args.heuristics_backward_survivor,
        heuristics_qubits_length=args.heuristics_qubits_length,
        ccx_cost=args.ccx_cost,
        cx_cost=args.cx_cost,
        x_cost=args.x_cost,
    )
    cpolicy = CounterPolicy()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    chunks: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    for start in range(1, T_max + 1, args.chunk_size):
        end = min(T_max, start + args.chunk_size - 1)
        cpath = chunk_path_for(workdir, args.n, start, end)
        data: dict[str, Any] | None = None

        if args.resume and not args.force and cpath.exists() and cpath.stat().st_size > 0:
            candidate = load_json(cpath)
            if args.ignore_policy_mismatch or _policy_matches(candidate, tpolicy=tpolicy, cpolicy=cpolicy):
                data = candidate
                print(f"[resume-chunk] {start}-{end}: ccx={data['ops'].get('ccx', 0)} cx={data['ops'].get('cx', 0)}", flush=True)
            else:
                print(f"[stale-chunk] {start}-{end}: policy mismatch, rebuilding from step checkpoints", flush=True)

        if data is None:
            print(
                f"[count-nct] {start}-{end} aux={args.aux_size} measurement={args.measurement_uncompute} "
                f"templates={args.templates} rounds={args.rounds} maxg={args.max_nct_segment_gates} "
                f"maxq={args.max_nct_segment_qubits} seg_timeout={args.segment_timeout_s}s "
                f"budget={args.optimization_budget_s}s step_timeout={args.step_timeout_s}s mode={args.timeout_mode}",
                flush=True,
            )
            tic = time.perf_counter()
            data = count_range(
                args.n,
                T_max,
                start,
                end,
                aux_size=args.aux_size,
                measurement_uncompute=args.measurement_uncompute,
                tpolicy=tpolicy,
                cpolicy=cpolicy,
                workdir=workdir,
                resume=args.resume,
                ignore_policy_mismatch=args.ignore_policy_mismatch,
                force=args.force,
                print_every=args.print_every,
                step_timeout_s=args.step_timeout_s,
                fallback_step_timeout_s=args.fallback_step_timeout_s,
                trace_step=args.trace_step,
            )
            _atomic_write_json(cpath, data)
            meta = data.get("optimization_meta", {})
            print(
                f"[done] {start}-{end}: {time.perf_counter() - tic:.2f}s "
                f"ccx={data['ops'].get('ccx', 0)} cx={data['ops'].get('cx', 0)} "
                f"qiskit_ok={meta.get('meta::nct_segments_qiskit_optimized', 0)} "
                f"changed={meta.get('meta::nct_segments_changed', 0)} "
                f"skip_q={meta.get('meta::nct_segments_skipped_many_qubits', 0)} "
                f"timeouts={meta.get('meta::nct_segments_skipped_timeout', 0)} "
                f"step_fallbacks={meta.get('meta::nct_step_timeout_fallback_unoptimized', 0)}",
                flush=True,
            )

        chunks.append(data)
        write_sum(out_path, n=args.n, T_max=T_max, chunks=chunks, elapsed_s=time.perf_counter() - t0, tpolicy=tpolicy, cpolicy=cpolicy)

    print(out_path)


if __name__ == "__main__":
    main()
