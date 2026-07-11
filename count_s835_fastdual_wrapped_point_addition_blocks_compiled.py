import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Optional

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

import eea_circuit_s835_fastdual as eea
from ccx_recursive_block_counter import CounterPolicy, count_gate_or_circuit, summarize_counter
from point_addition_fig14_s835_fastdual_wrapped_quadratic import build_point_addition_fig14_quadratic
from under1000_eea_shared_s835_fastdual_wrapped import SECP256K1_P, eea_forward_shared_instruction, shared_eea_layout
from quadratic_modular_arithmetic import (
    add_const_modp_instruction,
    neg_modp_instruction,
    ctrl_add_modp_instruction,
    ctrl_sub_modp_instruction,
    mul_zero_dbladd_instruction,
    mul_zero_dbladd_inverse_instruction,
    square_zero_dbladd_instruction,
    square_zero_dbladd_inverse_instruction,
    append_dbl_modp_quadratic,
    append_halve_modp_quadratic,
)

SECP256K1_GX = int("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798", 16)
SECP256K1_GY = int("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8", 16)
DEFAULT_EEA_MEASUREMENT_JSON = Path(__file__).with_name("eea_s835_fastdual_algorithm3_recursive_chunks_n256_measurement.json")
DEFAULT_EEA_STRICT_JSON = Path(__file__).with_name("eea_algorithm3_recursive_chunks_n256_checkpointed.json")


def _counter(d: Mapping[str, Any] | None = None) -> Counter:
    out = Counter()
    for k, v in (d or {}).items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[str(k)] += int(v)
    return out


def _scale(c: Counter, k: int) -> Counter:
    return Counter({name: int(value) * int(k) for name, value in c.items()})


def _jsonable_counter(c: Counter) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(c.items())}


def _summ(c: Counter, policy: CounterPolicy, *, note: Optional[str] = None, components: Optional[dict[str, int]] = None, compiled_from: Optional[str] = None) -> dict[str, Any]:
    out = summarize_counter(c, policy=policy)
    if note:
        out["note"] = note
    if components:
        out["components"] = {str(k): int(v) for k, v in components.items()}
    if compiled_from:
        out["compiled_from"] = compiled_from
    return out


def _qiskit_width_report(n: int, p: int, x2: int, y2: int, s_qubits: Optional[int]) -> dict[str, Any]:
    qc = build_point_addition_fig14_quadratic(n=n, p=p, x2=x2, y2=y2, s_qubits=s_qubits)
    return {
        "num_qubits": int(qc.num_qubits),
        "num_clbits": int(qc.num_clbits),
        "qregs": {reg.name: int(reg.size) for reg in qc.qregs},
        "cregs": {reg.name: int(reg.size) for reg in qc.cregs},
        "top_level_ops": {str(k): int(v) for k, v in qc.count_ops().items()},
        "has_only_ctrl_X_Y_A_S_qregs": [reg.name for reg in qc.qregs] == [
            "ctrl", "X_x1_to_x3", "Y_y1_to_y3", "A_shared_work", "S_shared_eea_arith"
        ],
        "has_extra_E_or_R_registers": any(reg.name.startswith(("E_", "R_")) for reg in qc.qregs),
    }


def _build_dbl_circuit(n: int, p: int) -> QuantumCircuit:
    acc = QuantumRegister(n, "acc")
    dirty = QuantumRegister(n, "dirty")
    aux = QuantumRegister(4, "aux4")
    m = ClassicalRegister(max(1, n), "m_arith")
    qc = QuantumCircuit(acc, dirty, aux, m, name=f"DBL_MODP_QUAD_COMPILED_{n}")
    append_dbl_modp_quadratic(qc, acc, dirty, aux, m, p=p)
    return qc


def _build_halve_circuit(n: int, p: int) -> QuantumCircuit:
    acc = QuantumRegister(n, "acc")
    dirty = QuantumRegister(n, "dirty")
    aux = QuantumRegister(4, "aux4")
    m = ClassicalRegister(max(1, n), "m_arith")
    qc = QuantumCircuit(acc, dirty, aux, m, name=f"HALVE_MODP_QUAD_COMPILED_{n}")
    append_halve_modp_quadratic(qc, acc, dirty, aux, m, p=p)
    return qc


def _count_actual(obj: Any, policy: CounterPolicy, cache: dict | None = None) -> Counter:
    # ccx_recursive_block_counter already has its own per-call cache.  We keep a
    # simple outer cache in the caller by block name, so this function is thin.
    return count_gate_or_circuit(obj, policy=policy)


def count_compiled_arithmetic_subblocks(n: int, p: int, x2: int, y2: int, policy: CounterPolicy) -> dict[str, Counter]:
    """Count every reusable arithmetic subblock from real Qiskit definitions."""
    blocks: dict[str, Counter] = {}

    # Constant blocks are counted with the actual constants used by Fig.14.
    blocks["add_const_x_minus_x2"] = _count_actual(add_const_modp_instruction(n, p, -x2, controlled=False, name="COMPILED_X_SUB_X2"), policy)
    blocks["cadd_const_y_minus_y2"] = _count_actual(add_const_modp_instruction(n, p, -y2, controlled=True, name="COMPILED_CTRL_Y_SUB_Y2"), policy)
    blocks["cadd_const_x_plus_3x2"] = _count_actual(add_const_modp_instruction(n, p, 3 * x2, controlled=True, name="COMPILED_CTRL_X_ADD_3X2"), policy)
    blocks["add_const_x_plus_x2"] = _count_actual(add_const_modp_instruction(n, p, x2, controlled=False, name="COMPILED_X_ADD_X2"), policy)
    blocks["cadd_const_y_minus_y2_final"] = blocks["cadd_const_y_minus_y2"].copy()
    blocks["cneg_modp"] = _count_actual(neg_modp_instruction(n, p, controlled=True, name="COMPILED_CTRL_NEG_X"), policy)

    # Hot repeated primitives counted from actual definitions.
    blocks["ctrl_add_modp"] = _count_actual(ctrl_add_modp_instruction(n, p), policy)
    blocks["ctrl_sub_modp"] = _count_actual(ctrl_sub_modp_instruction(n, p), policy)
    blocks["dbl_modp"] = _count_actual(_build_dbl_circuit(n, p), policy)
    blocks["halve_modp"] = _count_actual(_build_halve_circuit(n, p), policy)

    return blocks


def assemble_mul_square_from_compiled_primitives(blocks: dict[str, Counter], n: int) -> None:
    """Add MUL/SQUARE counters by multiplying compiled primitive subblock counts."""
    blocks["mul_zero_dbladd"] = _scale(blocks["ctrl_add_modp"], n) + _scale(blocks["dbl_modp"], n - 1)
    blocks["mul_zero_dbladd_inverse"] = _scale(blocks["ctrl_sub_modp"], n) + _scale(blocks["halve_modp"], n - 1)
    blocks["square_zero_dbladd"] = blocks["mul_zero_dbladd"].copy() + Counter({"cx": 2 * n})
    blocks["square_zero_dbladd_inverse"] = blocks["mul_zero_dbladd_inverse"].copy() + Counter({"cx": 2 * n})


def validate_full_mul_square(n: int, p: int, blocks: dict[str, Counter], policy: CounterPolicy) -> dict[str, Any]:
    """For small n, recursively count whole MUL/SQUARE definitions and compare."""
    actual_mul = _count_actual(mul_zero_dbladd_instruction(n, p), policy)
    actual_mul_inv = _count_actual(mul_zero_dbladd_inverse_instruction(n, p), policy)
    actual_square = _count_actual(square_zero_dbladd_instruction(n, p), policy)
    actual_square_inv = _count_actual(square_zero_dbladd_inverse_instruction(n, p), policy)
    tests = {
        "mul_zero_dbladd": (actual_mul, blocks["mul_zero_dbladd"]),
        "mul_zero_dbladd_inverse": (actual_mul_inv, blocks["mul_zero_dbladd_inverse"]),
        "square_zero_dbladd": (actual_square, blocks["square_zero_dbladd"]),
        "square_zero_dbladd_inverse": (actual_square_inv, blocks["square_zero_dbladd_inverse"]),
    }
    out: dict[str, Any] = {}
    ok = True
    for name, (actual, assembled) in tests.items():
        plus = actual - assembled
        minus = assembled - actual
        passed = not plus and not minus
        ok = ok and passed
        out[name] = {
            "passed": passed,
            "actual_recursive": _jsonable_counter(actual),
            "assembled_from_compiled_primitives": _jsonable_counter(assembled),
            "actual_minus_assembled": _jsonable_counter(plus),
            "assembled_minus_actual": _jsonable_counter(minus),
        }
    out["all_passed"] = ok
    return out


def _load_eea_alg3_counts(path: Optional[str], n: int, *, allow_n_mismatch: bool = False) -> tuple[Counter, dict[str, Any]]:
    chosen: Optional[Path] = Path(path) if path else None
    if chosen is None:
        if n == 256 and DEFAULT_EEA_MEASUREMENT_JSON.exists():
            chosen = DEFAULT_EEA_MEASUREMENT_JSON
        elif n == 256 and DEFAULT_EEA_STRICT_JSON.exists():
            chosen = DEFAULT_EEA_STRICT_JSON
        else:
            raise ValueError("No EEA JSON was supplied. Pass --eea-steps-json or generate one with run_eea_recursive_chunks_checkpoint.py.")
    data = json.loads(chosen.read_text(encoding="utf-8"))
    if (not allow_n_mismatch) and data.get("n") is not None and int(data.get("n")) != int(n):
        raise ValueError(f"EEA JSON n={data.get('n')} does not match requested n={n}. Pass --allow-eea-n-mismatch only for debugging.")
    ops = _counter(data.get("ops", data))
    meta = {
        "eea_step_mode": "loaded-recursive-json",
        "path": str(chosen),
        "source_mode": data.get("mode"),
        "n": data.get("n"),
        "T_max": data.get("T_max"),
        "num_qubits": data.get("num_qubits"),
        "range": data.get("range"),
        "chunks": data.get("chunks"),
        "measurement_based": any(ops.get(k, 0) for k in ["h", "measure", "reset", "cz"]),
    }
    return ops, meta


def count_compiled_eea_shared(n: int, p: int, policy: CounterPolicy, eea_steps_json: Optional[str], *, allow_n_mismatch: bool = False) -> tuple[Counter, dict[str, Any]]:
    """Count shared EEA block using real wrapper definition + loaded recursive steps."""
    layout = shared_eea_layout(n)
    skip_policy = CounterPolicy(
        mcx_policy=policy.mcx_policy,
        expand_swap_to_cx=policy.expand_swap_to_cx,
        skip_alg3_steps=True,
        stop_prefixes=policy.stop_prefixes,
    )
    g = eea_forward_shared_instruction(n, p)
    # The wrapped fastdual EEA step instructions are real definitions.  For n=256
    # we avoid expanding the 1476 step bodies here and instead load the chunk
    # counts obtained by recursively counting the same step circuits.  The
    # counter stops on the step instruction name and records STOP:: entries.
    stop_policy = CounterPolicy(
        mcx_policy=policy.mcx_policy,
        expand_swap_to_cx=policy.expand_swap_to_cx,
        skip_alg3_steps=False,
        stop_prefixes=("alg3_step_fastdual_wrapped_t", "alg3_step_fastdual_t"),
    )
    wrapper = count_gate_or_circuit(g, policy=stop_policy)
    stop_keys = [k for k in wrapper if str(k).startswith("STOP::alg3_step_fastdual")]
    skipped = sum(int(wrapper[k]) for k in stop_keys)
    for k in stop_keys:
        del wrapper[k]
    wrapper["meta::skipped_alg3_step"] += skipped
    if skipped != int(layout.T_max):
        raise RuntimeError(f"EEA wrapper stopped on {skipped} Algorithm-3 steps, expected {layout.T_max}.")
    alg3, alg3_meta = _load_eea_alg3_counts(eea_steps_json, n, allow_n_mismatch=allow_n_mismatch)
    full = wrapper + alg3
    meta = {
        "layout": layout.as_dict(),
        "eea_forward_gate_qubits": int(g.num_qubits),
        "skipped_alg3_steps_in_wrapper": skipped,
        "algorithm3_ops_loaded": _jsonable_counter(alg3),
        "shared_wrapper_overhead_compiled_recursive": _jsonable_counter(wrapper),
        **alg3_meta,
    }
    return full, meta


def fig15_outer_dynamic_overhead(n: int) -> Counter:
    return Counter({
        "h": n,
        "measure": n,
        "reset": n,
        "z": n,
        "cx": 3 * n,
        "meta::classically_controlled_z": n,
        "meta::swap_expanded_to_3cx": n,
    })


def assemble_fig14_fig15(blocks: dict[str, Counter], eea_forward: Counter, n: int) -> dict[str, Counter]:
    outer = fig15_outer_dynamic_overhead(n)
    mul = blocks["mul_zero_dbladd"]
    mul_inv = blocks["mul_zero_dbladd_inverse"]
    square = blocks["square_zero_dbladd"]
    square_inv = blocks["square_zero_dbladd_inverse"]

    # The real Fig.15 implementations contain two EEA-shaped blocks and three
    # multiplication-shaped blocks: mul, recompute mul, and inverse mul.
    idiv = _scale(eea_forward, 2) + _scale(mul, 2) + mul_inv + outer
    imul = _scale(eea_forward, 2) + _scale(mul, 2) + mul_inv + outer
    squ_minus = square + blocks["ctrl_sub_modp"] + square_inv

    point = Counter()
    point += blocks["add_const_x_minus_x2"]
    point += blocks["cadd_const_y_minus_y2"]
    point += idiv
    point += squ_minus
    point += blocks["cadd_const_x_plus_3x2"]
    point += imul
    point += blocks["cneg_modp"]
    point += blocks["add_const_x_plus_x2"]
    point += blocks["cadd_const_y_minus_y2_final"]

    return {
        "eea_forward_shared": eea_forward,
        "idiv_fig15": idiv,
        "imul_fig15": imul,
        "squ_minus": squ_minus,
        "point_addition_fig14_total": point,
    }


def resolve_constants(kind: str, x2: Optional[int], y2: Optional[int]) -> tuple[int, int, str]:
    if kind == "zero":
        return 0, 0, "zero"
    if kind == "secp256k1-generator":
        return SECP256K1_GX, SECP256K1_GY, "secp256k1-generator"
    if x2 is None or y2 is None:
        raise ValueError("--point-constant custom requires --x2 and --y2")
    return int(x2), int(y2), "custom"


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.perf_counter()
    x2, y2, const_kind = resolve_constants(args.point_constant, args.x2, args.y2)
    policy = CounterPolicy(mcx_policy=args.mcx_policy, expand_swap_to_cx=True)

    width = _qiskit_width_report(args.n, args.p, x2, y2, args.s_qubits)
    blocks = count_compiled_arithmetic_subblocks(args.n, args.p, x2, y2, policy)
    assemble_mul_square_from_compiled_primitives(blocks, args.n)

    eea_forward, eea_meta = count_compiled_eea_shared(args.n, args.p, policy, args.eea_steps_json, allow_n_mismatch=args.allow_eea_n_mismatch)
    derived = assemble_fig14_fig15(blocks, eea_forward, args.n)
    all_blocks = {**blocks, **derived}

    validation = None
    if args.validate_full_mul:
        if args.n > args.max_full_recursive_n:
            validation = {
                "skipped": True,
                "reason": f"n={args.n} exceeds --max-full-recursive-n={args.max_full_recursive_n}; full recursive MUL is too large."
            }
        else:
            validation = validate_full_mul_square(args.n, args.p, blocks, policy)

    summary_notes = {
        "ctrl_add_modp": "compiled recursively from ctrl_add_modp_instruction.definition",
        "ctrl_sub_modp": "compiled recursively from ctrl_sub_modp_instruction.definition",
        "dbl_modp": "compiled recursively from a real Qiskit dbl_modp circuit",
        "halve_modp": "compiled recursively from a real Qiskit halve_modp circuit",
        "mul_zero_dbladd": "assembled as n compiled ctrl_add_modp blocks plus n-1 compiled dbl_modp blocks",
        "mul_zero_dbladd_inverse": "assembled as n compiled ctrl_sub_modp blocks plus n-1 compiled halve_modp blocks",
        "square_zero_dbladd": "assembled like MUL plus 2n compiled CNOTs for copy/uncompute of the selected control bit",
        "square_zero_dbladd_inverse": "assembled like inverse MUL plus 2n compiled CNOTs for copy/uncompute of the selected control bit",
        "idiv_fig15": "assembled from compiled EEA/shared wrapper counts, compiled MUL counts, and explicit Fig.15 dynamic overhead",
        "imul_fig15": "assembled from compiled EEA/shared wrapper counts, compiled MUL counts, and explicit Fig.15 dynamic overhead",
    }
    components = {
        "mul_zero_dbladd": {"ctrl_add_modp": args.n, "dbl_modp": args.n - 1},
        "mul_zero_dbladd_inverse": {"ctrl_sub_modp": args.n, "halve_modp": args.n - 1},
        "square_zero_dbladd": {"ctrl_add_modp": args.n, "dbl_modp": args.n - 1, "copy_uncompute_cx": 2 * args.n},
        "square_zero_dbladd_inverse": {"ctrl_sub_modp": args.n, "halve_modp": args.n - 1, "copy_uncompute_cx": 2 * args.n},
        "squ_minus": {"square_zero_dbladd": 1, "ctrl_sub_modp": 1, "square_zero_dbladd_inverse": 1},
        "idiv_fig15": {"eea_forward_or_inverse": 2, "mul_zero_dbladd": 2, "mul_zero_dbladd_inverse": 1, "outer_h_measure_reset_z_swap": 1},
        "imul_fig15": {"eea_forward_or_inverse": 2, "mul_zero_dbladd": 2, "mul_zero_dbladd_inverse": 1, "outer_h_measure_reset_z_swap": 1},
    }

    return {
        "script": Path(__file__).name,
        "counting_mode": "S835_FASTDUAL_WRAPPED compiled-subblock-recursive + blockwise assembly",
        "important_note": (
            "This is not a closed-form arithmetic formula. Each reusable arithmetic primitive "
            "is built as a Qiskit circuit/Instruction and recursively counted. Repeated Horner "
            "blocks are assembled from those compiled primitive counts to avoid materializing "
            "hundreds of identical copies at n=256."
        ),
        "n": int(args.n),
        "p": int(args.p),
        "point_constant_kind": const_kind,
        "x2": int(x2),
        "y2": int(y2),
        "counter_policy": policy.as_dict(),
        "qiskit_width_report": width,
        "eea_meta": eea_meta,
        "block_summaries": {
            name: _summ(c, policy, note=summary_notes.get(name), components=components.get(name))
            for name, c in all_blocks.items()
        },
        "raw_block_counters": {name: _jsonable_counter(c) for name, c in all_blocks.items()},
        "key_ccx": {
            "ctrl_add_modp": int(blocks["ctrl_add_modp"].get("ccx", 0)),
            "dbl_modp": int(blocks["dbl_modp"].get("ccx", 0)),
            "mul_zero_dbladd": int(blocks["mul_zero_dbladd"].get("ccx", 0)),
            "square_zero_dbladd": int(blocks["square_zero_dbladd"].get("ccx", 0)),
            "eea_algorithm3_main_loop": int(_counter(eea_meta.get("algorithm3_ops_loaded", {})).get("ccx", 0)),
            "eea_forward_shared": int(eea_forward.get("ccx", 0)),
            "squ_minus": int(derived["squ_minus"].get("ccx", 0)),
            "idiv_fig15": int(derived["idiv_fig15"].get("ccx", 0)),
            "imul_fig15": int(derived["imul_fig15"].get("ccx", 0)),
            "point_addition_fig14_total": int(derived["point_addition_fig14_total"].get("ccx", 0)),
        },
        "validation": validation,
        "elapsed_s": time.perf_counter() - t0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compiled blockwise CCX/CNOT counter for the quadratic Fig.14 backend")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--p", type=lambda s: int(s, 0), default=SECP256K1_P)
    ap.add_argument("--s-qubits", type=int, default=None)
    ap.add_argument("--point-constant", choices=["secp256k1-generator", "zero", "custom"], default="secp256k1-generator")
    ap.add_argument("--x2", type=lambda s: int(s, 0), default=None)
    ap.add_argument("--y2", type=lambda s: int(s, 0), default=None)
    ap.add_argument("--eea-steps-json", default=None, help="recursive Algorithm-3 EEA chunks JSON; defaults to measurement JSON if present")
    ap.add_argument("--allow-eea-n-mismatch", action="store_true", help="debug only: allow loading an EEA JSON whose n differs from --n")
    ap.add_argument("--mcx-policy", choices=["clean-vchain", "keep"], default="clean-vchain")
    ap.add_argument("--validate-full-mul", action="store_true", help="for small n, recursively count whole MUL/SQUARE definitions and compare")
    ap.add_argument("--max-full-recursive-n", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    report = build_report(args)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
