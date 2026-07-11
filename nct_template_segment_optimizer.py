"""Bounded fail-open NCT-segment template optimizer and recursive counter.

Drop-in replacement for ``nct_template_segment_optimizer.py``.

It applies Qiskit's TemplateOptimization only to pure reversible NCT regions
(``x``, ``cx``, ``ccx``).  Measurement/reset/feed-forward and other non-NCT
operations are counted as boundaries.

Safeguards for n=256 S835 runs:

* ``rounds <= 0`` really disables TemplateOptimization and counts unchanged.
* Pure-NCT runs are split into small windows.
* Each window is compacted onto only the qubits it actually touches before
  Qiskit sees it.  This avoids matching templates on a 1000+ qubit parent
  circuit with hundreds of idle qubits.
* Windows touching too many active qubits are counted unchanged.
* Each Qiskit template attempt has a timeout.
* Each top-level count has an attempt/time budget.  After timeout/exception or
  budget exhaustion, the remaining windows are counted unchanged.
* Counts remain complete: skipped windows still contribute their unchanged NCT
  gates to the final CCX/Toffoli and CX totals.
"""
import importlib
import multiprocessing as _mp
import pkgutil
import signal
import sys
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Iterable, Sequence

try:
    from qiskit import QuantumCircuit
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import TemplateOptimization
except Exception:  # pragma: no cover
    QuantumCircuit = None  # type: ignore
    PassManager = None  # type: ignore
    TemplateOptimization = None  # type: ignore

from ccx_recursive_block_counter import (
    CounterPolicy,
    is_mcx,
    iter_items,
    mcx_as_counter,
    summarize_counter,
)

NCT = {"x", "cx", "ccx"}
PRIMITIVE_DYNAMIC = {
    "h", "z", "cz", "measure", "reset", "barrier", "delay", "id",
    "s", "sdg", "t", "tdg", "sx", "sxdg", "rx", "ry", "rz", "p", "u", "u1", "u2", "u3",
}
CONTROL_FLOW = {"if_else", "for_loop", "while_loop", "switch_case"}

# Compact representation of one NCT operation: (gate_name, parent_qubit_indices).
NCTOp = tuple[str, tuple[int, ...]]


@dataclass(frozen=True)
class NCTTemplatePolicy:
    templates: str = "small-nct"
    rounds: int = 1
    ccx_cost: int = 100
    cx_cost: int = 1
    x_cost: int = 1

    # Local window controls.  For n=256, large windows can have many active
    # qubits and Qiskit template matching becomes combinatorial.
    max_segment_gates: int = 12
    min_segment_gates: int = 2
    max_segment_qubits: int = 6

    heuristics_backward_length: int = 3
    heuristics_backward_survivor: int = 1
    heuristics_qubits_length: int = 1

    # Per compacted-window TemplateOptimization timeout.  On timeout, the window
    # is counted unchanged and the run continues.
    segment_timeout_s: float = 1.0
    timeout_mode: str = "auto"  # auto|signal|process|none

    # Top-level throttles, applied per count_gate_or_circuit_nct_optimized call.
    optimization_budget_s: float = 20.0
    max_template_attempts_per_count: int = 80
    disable_templates_after_timeout: bool = True
    disable_templates_after_exception: bool = False

    # Fast, deterministic adjacent self-inverse cancellation.  This is safe and
    # non-search-based.  Fallback rounds=0 should set this False if a completely
    # unchanged count is desired.
    fast_cancel: bool = True

    progress: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_qiskit() -> None:
    if QuantumCircuit is None or PassManager is None or TemplateOptimization is None:
        raise RuntimeError("Qiskit with TemplateOptimization is required for NCT optimization.")


def _local_nct_templates() -> tuple[Any, ...]:
    """Small identity templates that are safe fallbacks across Qiskit versions."""
    _require_qiskit()
    out: list[Any] = []

    q1 = QuantumCircuit(1, name="x_x_identity")
    q1.x(0)
    q1.x(0)
    out.append(q1)

    q2 = QuantumCircuit(2, name="cx_cx_identity")
    q2.cx(0, 1)
    q2.cx(0, 1)
    out.append(q2)

    q3 = QuantumCircuit(3, name="ccx_ccx_identity")
    q3.ccx(0, 1, 2)
    q3.ccx(0, 1, 2)
    out.append(q3)

    return tuple(out)


@lru_cache(maxsize=None)
def load_nct_templates(kind: str = "small-nct") -> tuple[Any, ...]:
    """Load Qiskit's built-in NCT templates plus tiny local identity templates."""
    _require_qiskit()
    if kind not in {"small-nct", "all-nct"}:
        raise ValueError(f"unknown template kind {kind!r}; use small-nct or all-nct")

    templates: list[Any] = []
    try:
        import qiskit.circuit.library.templates.nct as nct_pkg  # type: ignore
        for modinfo in pkgutil.iter_modules(nct_pkg.__path__):
            name = modinfo.name
            if not name.startswith("template_nct_"):
                continue
            try:
                mod = importlib.import_module(f"qiskit.circuit.library.templates.nct.{name}")
                circ = getattr(mod, name)()
            except Exception:
                continue
            ops = circ.count_ops()
            if any(str(k).lower() not in NCT for k in ops):
                continue
            size = sum(int(v) for v in ops.values())
            nq = int(circ.num_qubits)
            if kind == "small-nct":
                if nq <= 4 and size <= 12:
                    templates.append(circ)
            else:
                templates.append(circ)
    except Exception:
        pass

    templates.extend(_local_nct_templates())
    if not templates:
        raise RuntimeError("No NCT templates were available.")
    return tuple(templates)


def nct_cost(ops: Counter, policy: NCTTemplatePolicy) -> int:
    opaque_penalty = 10**12 * sum(
        int(v)
        for k, v in ops.items()
        if str(k) not in NCT and not str(k).startswith("meta::")
    )
    return (
        int(ops.get("ccx", 0)) * int(policy.ccx_cost)
        + int(ops.get("cx", 0)) * int(policy.cx_cost)
        + int(ops.get("x", 0)) * int(policy.x_cost)
        + opaque_penalty
    )


def _count_ops(ops: Sequence[NCTOp]) -> Counter:
    out: Counter = Counter()
    for name, _qidx in ops:
        out[name] += 1
    return out


def _count_direct_nct_circuit(qc: Any) -> Counter:
    out: Counter = Counter()
    for inst, _qargs, _cargs in iter_items(qc):
        name = str(getattr(inst, "name", "")).lower()
        if name in NCT:
            out[name] += 1
        else:
            out[f"OPAQUE::{name or 'unnamed'}"] += 1
    return out


def _active_qubits(ops: Sequence[NCTOp]) -> tuple[int, ...]:
    active: set[int] = set()
    for _name, qidx in ops:
        active.update(int(q) for q in qidx)
    return tuple(sorted(active))


def _build_compact_segment(ops: Sequence[NCTOp]) -> Any:
    """Build a compact Qiskit circuit using only active qubits in this window."""
    _require_qiskit()
    active = _active_qubits(ops)
    mapping = {q: i for i, q in enumerate(active)}
    qc = QuantumCircuit(max(1, len(active)), name="nct_window_compact")
    for name, qidx in ops:
        local = [mapping[int(q)] for q in qidx]
        if name == "x":
            qc.x(local[0])
        elif name == "cx":
            qc.cx(local[0], local[1])
        elif name == "ccx":
            qc.ccx(local[0], local[1], local[2])
        else:
            raise ValueError(f"not an NCT instruction: {name}")
    return qc


def _fast_cancel_ops(ops: Sequence[NCTOp]) -> list[NCTOp]:
    """Cancel adjacent identical self-inverse NCT gates."""
    stack: list[NCTOp] = []
    for op in ops:
        if stack and stack[-1] == op:
            stack.pop()
        else:
            stack.append(op)
    return stack


class _TemplateTimeout(Exception):
    pass


def _make_template_pass(templates: Sequence[Any], policy: NCTTemplatePolicy) -> Any:
    _require_qiskit()
    try:
        opt = TemplateOptimization(
            template_list=list(templates),
            heuristics_backward_param=[
                int(policy.heuristics_backward_length),
                int(policy.heuristics_backward_survivor),
            ],
            heuristics_qubits_param=[int(policy.heuristics_qubits_length)],
            user_cost_dict={
                "x": int(policy.x_cost),
                "cx": int(policy.cx_cost),
                "ccx": int(policy.ccx_cost),
            },
        )
    except TypeError:
        # Older Qiskit releases do not accept all keyword arguments.
        opt = TemplateOptimization(template_list=list(templates))
    return PassManager([opt])


def _apply_template_counts_no_timeout(segment: Any, templates: Sequence[Any], policy: NCTTemplatePolicy) -> Counter:
    pm = _make_template_pass(templates, policy)
    cur = segment
    for _ in range(int(policy.rounds)):
        cur = pm.run(cur)
    return _count_direct_nct_circuit(cur)


def _signal_timeout_supported() -> bool:
    return (
        hasattr(signal, "SIGALRM")
        and hasattr(signal, "setitimer")
        and threading.current_thread() is threading.main_thread()
    )


def _resolve_timeout_mode(mode: str) -> str:
    mode = (mode or "auto").lower()
    if mode not in {"auto", "signal", "process", "none"}:
        raise ValueError("timeout_mode must be one of: auto, signal, process, none")
    if mode == "auto":
        return "signal" if _signal_timeout_supported() else "process"
    return mode


def _apply_template_counts_signal(segment: Any, templates: Sequence[Any], policy: NCTTemplatePolicy) -> tuple[Counter | None, str, str | None]:
    timeout_s = float(policy.segment_timeout_s)
    if timeout_s <= 0:
        try:
            return _apply_template_counts_no_timeout(segment, templates, policy), "ok", None
        except Exception as exc:
            return None, "exception", repr(exc)
    if not _signal_timeout_supported():
        return None, "unsupported_timeout", "SIGALRM/setitimer unavailable outside POSIX main thread"

    old_handler = signal.getsignal(signal.SIGALRM)

    def _handler(_signum: int, _frame: Any) -> None:
        raise _TemplateTimeout()

    try:
        signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_s)
        try:
            return _apply_template_counts_no_timeout(segment, templates, policy), "ok", None
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
    except _TemplateTimeout:
        return None, "timeout", None
    except Exception as exc:
        return None, "exception", repr(exc)
    finally:
        try:
            signal.signal(signal.SIGALRM, old_handler)
        except Exception:
            pass


def _best_mp_context() -> Any:
    try:
        methods = _mp.get_all_start_methods()
        if "fork" in methods and sys.platform != "win32":
            return _mp.get_context("fork")
    except Exception:
        pass
    return _mp.get_context("spawn")


def _template_worker(conn: Any, segment: Any, templates: Sequence[Any], policy: NCTTemplatePolicy) -> None:  # pragma: no cover
    try:
        after = _apply_template_counts_no_timeout(segment, templates, policy)
        conn.send(("ok", {str(k): int(v) for k, v in after.items()}, None))
    except Exception as exc:
        try:
            conn.send(("exception", None, repr(exc)))
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _apply_template_counts_process(segment: Any, templates: Sequence[Any], policy: NCTTemplatePolicy) -> tuple[Counter | None, str, str | None]:
    timeout_s = float(policy.segment_timeout_s)
    if timeout_s <= 0:
        try:
            return _apply_template_counts_no_timeout(segment, templates, policy), "ok", None
        except Exception as exc:
            return None, "exception", repr(exc)

    parent = child = proc = None
    try:
        ctx = _best_mp_context()
        parent, child = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=_template_worker, args=(child, segment, tuple(templates), policy))
        proc.daemon = True
        proc.start()
        child.close()
        child = None
        if parent.poll(timeout_s):
            status, payload, err = parent.recv()
            proc.join(timeout=1.0)
            if status == "ok":
                return Counter({str(k): int(v) for k, v in payload.items()}), "ok", None
            return None, str(status), err
        proc.terminate()
        proc.join(timeout=1.0)
        if proc.is_alive():
            try:
                proc.kill()
            except Exception:
                pass
            proc.join(timeout=1.0)
        return None, "timeout", None
    except Exception as exc:
        return None, "exception", repr(exc)
    finally:
        for obj in (parent, child):
            try:
                if obj is not None:
                    obj.close()
            except Exception:
                pass


def _apply_template_counts_with_timeout(segment: Any, templates: Sequence[Any], policy: NCTTemplatePolicy) -> tuple[Counter | None, str, str | None]:
    mode = _resolve_timeout_mode(policy.timeout_mode)
    if mode == "none":
        try:
            return _apply_template_counts_no_timeout(segment, templates, policy), "ok", None
        except Exception as exc:
            return None, "exception", repr(exc)
    if mode == "signal":
        return _apply_template_counts_signal(segment, templates, policy)
    if mode == "process":
        return _apply_template_counts_process(segment, templates, policy)
    raise AssertionError(mode)


def _new_state() -> dict[str, Any]:
    return {
        "attempts": 0,
        "failures": 0,
        "disabled": False,
        "disabled_reason": "",
        "started_s": time.perf_counter(),
    }


def _mark_changed_meta(before: Counter, after: Counter, policy: NCTTemplatePolicy, meta: dict[str, int]) -> None:
    if after != before:
        meta["nct_segments_changed"] = 1
        meta["nct_cost_saved"] = max(0, nct_cost(before, policy) - nct_cost(after, policy))
        meta["nct_ccx_saved"] = int(before.get("ccx", 0)) - int(after.get("ccx", 0))
        meta["nct_cx_saved"] = int(before.get("cx", 0)) - int(after.get("cx", 0))
        meta["nct_x_saved"] = int(before.get("x", 0)) - int(after.get("x", 0))


def _run_template_optimization_ops(
    ops: Sequence[NCTOp],
    *,
    templates: Sequence[Any],
    policy: NCTTemplatePolicy,
    state: dict[str, Any],
) -> tuple[Counter, Counter, dict[str, int]]:
    """Optimize one compact pure-NCT window and return before/after/meta."""
    before = _count_ops(ops)
    gate_count = sum(before.values())

    if gate_count == 0:
        return before, before, {}
    if int(policy.rounds) <= 0:
        return before, before, {"nct_segments_skipped_rounds_zero": 1}
    if gate_count < int(policy.min_segment_gates):
        return before, before, {"nct_segments_skipped_short": 1}

    working_ops = list(ops)
    after_fast = before
    meta: dict[str, int] = {}
    if bool(policy.fast_cancel):
        working_ops = _fast_cancel_ops(working_ops)
        after_fast = _count_ops(working_ops)
        if after_fast != before:
            meta["nct_segments_fast_cancel_changed"] = 1
            _mark_changed_meta(before, after_fast, policy, meta)

    # If fast cancellation removed everything, no need to invoke Qiskit.
    if not working_ops:
        meta["nct_segments_optimized_fast_only"] = 1
        return before, after_fast, meta

    active_q = len(_active_qubits(working_ops))
    if int(policy.max_segment_qubits) > 0 and active_q > int(policy.max_segment_qubits):
        meta["nct_segments_skipped_many_qubits"] = 1
        return before, after_fast, meta
    if not templates:
        meta["nct_segments_skipped_no_templates"] = 1
        return before, after_fast, meta

    if state.get("disabled", False):
        reason = str(state.get("disabled_reason") or "disabled")
        meta[f"nct_segments_skipped_{reason}"] = 1
        return before, after_fast, meta

    budget_s = float(policy.optimization_budget_s or 0.0)
    if budget_s > 0 and (time.perf_counter() - float(state.get("started_s", time.perf_counter()))) >= budget_s:
        state["disabled"] = True
        state["disabled_reason"] = "budget_exhausted"
        meta["nct_segments_skipped_budget_exhausted"] = 1
        meta["nct_optimizer_disabled_budget_exhausted"] = 1
        return before, after_fast, meta

    max_attempts = int(policy.max_template_attempts_per_count or 0)
    if max_attempts > 0 and int(state.get("attempts", 0)) >= max_attempts:
        state["disabled"] = True
        state["disabled_reason"] = "attempt_limit"
        meta["nct_segments_skipped_attempt_limit"] = 1
        meta["nct_optimizer_disabled_attempt_limit"] = 1
        return before, after_fast, meta

    state["attempts"] = int(state.get("attempts", 0)) + 1
    meta["nct_template_attempts"] = 1

    try:
        segment = _build_compact_segment(working_ops)
    except Exception:
        state["failures"] = int(state.get("failures", 0)) + 1
        meta["nct_segments_skipped_build_exception"] = 1
        return before, after_fast, meta

    after_qiskit, status, err = _apply_template_counts_with_timeout(segment, templates, policy)
    if status == "timeout":
        state["failures"] = int(state.get("failures", 0)) + 1
        if bool(policy.disable_templates_after_timeout):
            state["disabled"] = True
            state["disabled_reason"] = "after_timeout"
            meta["nct_optimizer_disabled_after_timeout"] = 1
        meta["nct_segments_skipped_timeout"] = 1
        return before, after_fast, meta
    if status == "unsupported_timeout":
        state["failures"] = int(state.get("failures", 0)) + 1
        meta["nct_segments_skipped_timeout_unsupported"] = 1
        return before, after_fast, meta
    if status != "ok" or after_qiskit is None:
        state["failures"] = int(state.get("failures", 0)) + 1
        if bool(policy.disable_templates_after_exception):
            state["disabled"] = True
            state["disabled_reason"] = "after_exception"
            meta["nct_optimizer_disabled_after_exception"] = 1
        meta["nct_segments_skipped_exception"] = 1
        if err:
            meta["nct_segments_skipped_exception_with_message"] = 1
        return before, after_fast, meta

    meta["nct_segments_qiskit_optimized"] = 1
    if nct_cost(after_qiskit, policy) <= nct_cost(after_fast, policy):
        _mark_changed_meta(before, after_qiskit, policy, meta)
        return before, after_qiskit, meta

    meta["nct_segments_rejected_cost_increase"] = 1
    return before, after_fast, meta


def _definition_cache_key(inst: Any) -> tuple[Any, ...]:
    return (
        getattr(inst, "name", "<unnamed>"),
        int(getattr(inst, "num_qubits", 0)),
        int(getattr(inst, "num_clbits", 0)),
        id(getattr(inst, "definition", None)),
    )


def _has_classical_condition(inst: Any, cargs: Sequence[Any] | None = None) -> bool:
    if cargs:
        return True
    if getattr(inst, "condition", None) is not None:
        return True
    try:
        return bool(getattr(inst, "condition_bits", []))
    except Exception:
        return False


def count_circuit_nct_optimized(
    qc: Any,
    *,
    counter_policy: CounterPolicy | None = None,
    template_policy: NCTTemplatePolicy | None = None,
    cache: dict[tuple[Any, ...], Counter] | None = None,
    state: dict[str, Any] | None = None,
) -> Counter:
    if counter_policy is None:
        counter_policy = CounterPolicy()
    if template_policy is None:
        template_policy = NCTTemplatePolicy()
    if cache is None:
        cache = {}
    if state is None:
        state = _new_state()

    total: Counter = Counter()
    if int(template_policy.rounds) <= 0:
        templates: tuple[Any, ...] = ()
    else:
        try:
            templates = load_nct_templates(template_policy.templates)
        except Exception:
            templates = ()
            total["meta::nct_template_load_failed"] += 1

    seg: list[NCTOp] = []

    def _add_window_result(before: Counter, after: Counter, meta: dict[str, int]) -> None:
        total.update(after)
        for k, v in meta.items():
            total[f"meta::{k}"] += int(v)
        total["meta::nct_gates_before"] += int(sum(before.values()))
        total["meta::nct_gates_after"] += int(sum(after.values()))

    def flush() -> None:
        nonlocal seg, total
        if not seg:
            return
        max_g = int(template_policy.max_segment_gates)
        if max_g <= 0 or len(seg) <= max_g:
            windows: Iterable[Sequence[NCTOp]] = (tuple(seg),)
        else:
            windows = (tuple(seg[i:i + max_g]) for i in range(0, len(seg), max_g))
        for window in windows:
            before, after, meta = _run_template_optimization_ops(
                window,
                templates=templates,
                policy=template_policy,
                state=state,
            )
            _add_window_result(before, after, meta)
        seg = []

    for inst, qargs, cargs in iter_items(qc):
        name = str(getattr(inst, "name", "")).lower()
        if name in NCT and not _has_classical_condition(inst, cargs):
            try:
                qidx = tuple(int(qc.find_bit(q).index) for q in qargs)
            except Exception:
                flush()
                total[name] += 1
                continue
            seg.append((name, qidx))
            continue

        flush()
        total += count_instruction_nct_optimized(
            inst,
            counter_policy=counter_policy,
            template_policy=template_policy,
            cache=cache,
            state=state,
        )

    flush()
    if template_policy.progress:
        total["meta::nct_template_attempts_state"] += int(state.get("attempts", 0))
        total["meta::nct_template_failures_state"] += int(state.get("failures", 0))
    return total


def count_instruction_nct_optimized(
    inst: Any,
    *,
    counter_policy: CounterPolicy,
    template_policy: NCTTemplatePolicy,
    cache: dict[tuple[Any, ...], Counter],
    state: dict[str, Any],
) -> Counter:
    name = str(getattr(inst, "name", "<unnamed>")).lower()

    if getattr(counter_policy, "skip_alg3_steps", False) and name.startswith("alg3_step"):
        return Counter({"meta::skipped_alg3_step": 1})

    for pref in counter_policy.stop_prefixes:
        if name.startswith(str(pref).lower()):
            return Counter({f"STOP::{name}": 1})

    # If an NCT gate reached here, it was classically controlled or otherwise not
    # safe to include in a pure reversible window.  Count it, but do not rewrite it.
    if name in NCT:
        return Counter({name: 1})

    if is_mcx(inst):
        return mcx_as_counter(inst, counter_policy)

    if name == "swap":
        if counter_policy.expand_swap_to_cx:
            return Counter({"cx": 3, "meta::swap_expanded_to_3cx": 1})
        return Counter({"swap": 1})

    if name in PRIMITIVE_DYNAMIC:
        return Counter({name: 1})

    if name in CONTROL_FLOW and hasattr(inst, "blocks"):
        out = Counter({f"meta::control_flow_{name}": 1})
        for block in inst.blocks:
            out += count_circuit_nct_optimized(
                block,
                counter_policy=counter_policy,
                template_policy=template_policy,
                cache=cache,
                state=state,
            )
        return out

    definition = getattr(inst, "definition", None)
    if definition is None:
        return Counter({f"OPAQUE::{name}": 1})

    key = _definition_cache_key(inst)
    if key in cache:
        cached = cache[key].copy()
        cached["meta::cache_hit"] += 1
        return cached

    out = count_circuit_nct_optimized(
        definition,
        counter_policy=counter_policy,
        template_policy=template_policy,
        cache=cache,
        state=state,
    )
    cache[key] = out.copy()
    return out


def count_gate_or_circuit_nct_optimized(
    obj: Any,
    *,
    counter_policy: CounterPolicy | None = None,
    template_policy: NCTTemplatePolicy | None = None,
) -> Counter:
    if counter_policy is None:
        counter_policy = CounterPolicy()
    if template_policy is None:
        template_policy = NCTTemplatePolicy()
    if hasattr(obj, "data"):
        return count_circuit_nct_optimized(obj, counter_policy=counter_policy, template_policy=template_policy)
    definition = getattr(obj, "definition", None)
    if definition is None:
        return Counter({f"OPAQUE::{getattr(obj, 'name', '<unnamed>').lower()}": 1})
    return count_circuit_nct_optimized(definition, counter_policy=counter_policy, template_policy=template_policy)


def summarize_nct_optimized_counter(ops: Counter, *, counter_policy: CounterPolicy, template_policy: NCTTemplatePolicy) -> dict[str, Any]:
    out = summarize_counter(ops, policy=counter_policy)
    out["nct_template_policy"] = template_policy.as_dict()
    out["nct_optimization_meta"] = {
        k: int(v)
        for k, v in sorted(ops.items())
        if str(k).startswith("meta::nct_")
    }
    return out
