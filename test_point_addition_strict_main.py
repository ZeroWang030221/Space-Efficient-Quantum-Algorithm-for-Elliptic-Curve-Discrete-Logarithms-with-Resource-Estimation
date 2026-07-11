import argparse
import json
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterable, Sequence


def _require_qiskit_or_skip() -> None:
    try:
        import qiskit  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment-dependent preflight
        print(f"[SKIP] Qiskit is not installed, so point-addition circuit tests were skipped: {exc}")
        raise SystemExit(0)


_require_qiskit_or_skip()

from ccx_recursive_block_counter import CounterPolicy  # noqa: E402
from count_s835_fastdual_wrapped_point_addition_blocks_compiled import (  # noqa: E402
    build_report,
    count_compiled_arithmetic_subblocks,
    validate_full_mul_square,
    assemble_mul_square_from_compiled_primitives,
)
from point_addition_fig14_s835_fastdual_wrapped_quadratic import (  # noqa: E402
    SECP256K1_P,
    build_point_addition_fig14_quadratic,
)
from under1000_eea_shared_s835_fastdual_wrapped import (  # noqa: E402
    shared_eea_layout,
    shared_eea_s_qubits,
)


@dataclass(frozen=True)
class TestItem:
    name: str
    fn: Callable[[], None]


def _items(qc):
    for item in qc.data:
        if hasattr(item, "operation"):
            yield item.operation, tuple(item.qubits), tuple(item.clbits)
        else:  # qiskit < 1.0 compatibility
            yield item


def _op_names(qc) -> list[str]:
    return [getattr(inst, "name", "").lower() for inst, _qargs, _cargs in _items(qc)]


def _qreg_sizes(qc) -> dict[str, int]:
    return {reg.name: int(reg.size) for reg in qc.qregs}


def _creg_sizes(qc) -> dict[str, int]:
    return {reg.name: int(reg.size) for reg in qc.cregs}


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _first_index_containing(names: Sequence[str], needle: str) -> int:
    needle = needle.lower()
    for i, name in enumerate(names):
        if needle in name:
            return i
    raise AssertionError(f"missing operation containing {needle!r}; first operations: {list(names[:40])}")


def _assert_no_opaque_counter_terms(counter: Counter, label: str) -> None:
    opaque = {k: int(v) for k, v in counter.items() if str(k).startswith("OPAQUE::") and int(v) != 0}
    if opaque:
        raise AssertionError(f"{label} contains opaque recursive counter terms: {opaque}")


def test_s835_width_and_register_layout(*, small_n: int, small_p: int, skip_n256: bool) -> None:
    qc = build_point_addition_fig14_quadratic(n=small_n, p=small_p, x2=2 % small_p, y2=3 % small_p)
    s = shared_eea_s_qubits(small_n)
    _assert(qc.num_qubits == 1 + 3 * small_n + s, f"small-n quantum width mismatch: got {qc.num_qubits}, expected {1 + 3 * small_n + s}")
    _assert(_qreg_sizes(qc) == {
        "ctrl": 1,
        "X_x1_to_x3": small_n,
        "Y_y1_to_y3": small_n,
        "A_shared_work": small_n,
        "S_shared_eea_arith": s,
    }, f"unexpected small-n qregs: {_qreg_sizes(qc)}")
    _assert(_creg_sizes(qc) == {
        "b_div": small_n,
        "b_mul": small_n,
        "m_arith": max(1, small_n),
    }, f"unexpected small-n cregs: {_creg_sizes(qc)}")
    _assert(not any(name.startswith(("E_", "R_")) for name in _qreg_sizes(qc)), "point-addition circuit must not allocate separate E_* or R_* registers")

    try:
        build_point_addition_fig14_quadratic(n=small_n, p=small_p, x2=0, y2=0, s_qubits=s - 1)
    except ValueError:
        pass
    else:
        raise AssertionError("building with one fewer S qubit should fail")

    if not skip_n256:
        qc256 = build_point_addition_fig14_quadratic(n=256, p=SECP256K1_P, x2=0, y2=0)
        _assert(shared_eea_s_qubits(256) == 66, f"S835 layout requires S=66 at n=256, got S={shared_eea_s_qubits(256)}")
        _assert(qc256.num_qubits == 835, f"n=256 wrapped point addition width got {qc256.num_qubits}, expected 835")
        _assert(_qreg_sizes(qc256)["S_shared_eea_arith"] == 66, f"n=256 S register size mismatch: {_qreg_sizes(qc256)}")


def test_fig14_top_level_schedule(*, small_n: int, small_p: int) -> None:
    qc = build_point_addition_fig14_quadratic(n=small_n, p=small_p, x2=2 % small_p, y2=3 % small_p)
    names = _op_names(qc)
    counts = Counter(names)

    _assert(names[0].startswith("quad_x_sub_x2"), f"first Fig.14 op should subtract x2 from X, got {names[0]!r}")
    _assert(names[1].startswith("quad_ctrl_y_sub_y2"), f"second Fig.14 op should controlled-subtract y2 from Y, got {names[1]!r}")
    _assert(names[-3].startswith("quad_ctrl_neg_x"), f"third-from-last op should controlled-negate X, got {names[-3]!r}")
    _assert(names[-2].startswith("quad_x_add_x2"), f"second-from-last op should add x2 back to X, got {names[-2]!r}")
    _assert(names[-1].startswith("quad_ctrl_y_sub_y2_final"), f"last op should final controlled-subtract y2 from Y, got {names[-1]!r}")

    required_substrings = [
        "quad_x_sub_x2",
        "quad_ctrl_y_sub_y2",
        "eea_forward_shared_alg3_fastdual_wrapped",
        "mul_zero_dbladd_quad",
        "square_zero_dbladd_quad",
        "ctrl_sub_modp_quad",
        "quad_ctrl_x_add_3x2",
        "quad_ctrl_neg_x",
        "quad_x_add_x2",
        "quad_ctrl_y_sub_y2_final",
    ]
    for needle in required_substrings:
        _first_index_containing(names, needle)

    ordered = [
        "quad_x_sub_x2",
        "quad_ctrl_y_sub_y2",
        "eea_forward_shared_alg3_fastdual_wrapped",
        "square_zero_dbladd_quad",
        "ctrl_sub_modp_quad",
        "quad_ctrl_x_add_3x2",
        "quad_ctrl_neg_x",
        "quad_x_add_x2",
        "quad_ctrl_y_sub_y2_final",
    ]
    positions = [_first_index_containing(names, needle) for needle in ordered]
    _assert(positions == sorted(positions), f"Fig.14 major blocks are out of order: {list(zip(ordered, positions))}")

    # Fig.15 division and multiplication each H-measure-reset the Y register and swap Y/A once.
    _assert(counts.get("h", 0) == 2 * small_n, f"expected {2 * small_n} top-level H operations, got {counts.get('h', 0)}")
    _assert(counts.get("measure", 0) == 2 * small_n, f"expected {2 * small_n} top-level measurements, got {counts.get('measure', 0)}")
    _assert(counts.get("reset", 0) == 2 * small_n, f"expected {2 * small_n} top-level resets, got {counts.get('reset', 0)}")
    _assert(counts.get("swap", 0) == 2 * small_n, f"expected {2 * small_n} top-level Y/A swaps, got {counts.get('swap', 0)}")
    z_or_if = counts.get("z", 0) + counts.get("if_else", 0)
    _assert(z_or_if >= 2 * small_n, f"expected at least {2 * small_n} classically controlled Z corrections, got z+if_else={z_or_if}")


def test_compiled_arithmetic_block_assembly(*, small_n: int, small_p: int) -> None:
    policy = CounterPolicy(mcx_policy="clean-vchain", expand_swap_to_cx=True)
    blocks = count_compiled_arithmetic_subblocks(small_n, small_p, x2=2 % small_p, y2=3 % small_p, policy=policy)
    assemble_mul_square_from_compiled_primitives(blocks, small_n)

    required_blocks = [
        "add_const_x_minus_x2",
        "cadd_const_y_minus_y2",
        "cadd_const_x_plus_3x2",
        "add_const_x_plus_x2",
        "cneg_modp",
        "ctrl_add_modp",
        "ctrl_sub_modp",
        "dbl_modp",
        "halve_modp",
        "mul_zero_dbladd",
        "mul_zero_dbladd_inverse",
        "square_zero_dbladd",
        "square_zero_dbladd_inverse",
    ]
    missing = [name for name in required_blocks if name not in blocks]
    _assert(not missing, f"compiled block counter is missing blocks: {missing}")
    for name, counter in blocks.items():
        _assert_no_opaque_counter_terms(counter, name)

    validation = validate_full_mul_square(small_n, small_p, blocks, policy)
    if not validation.get("all_passed"):
        failed = {k: v for k, v in validation.items() if isinstance(v, dict) and not v.get("passed", True)}
        raise AssertionError("compiled MUL/SQUARE assembly does not match recursive definitions: " + json.dumps(failed, indent=2, sort_keys=True))


def test_tiny_point_addition_counter_report(*, small_n: int, small_p: int) -> None:
    layout = shared_eea_layout(small_n)
    # The point-addition counter consumes Algorithm-3 counts from JSON.  This tiny
    # synthetic file keeps this integration test fast while still checking that
    # the wrapper stops on exactly T_max real Algorithm-3 step instructions and
    # that the report assembly path is wired correctly.
    synthetic_eea = {
        "mode": "unit-test-synthetic-algorithm3-counts",
        "n": int(small_n),
        "T_max": int(layout.T_max),
        "num_qubits": int(4 + 2 * (small_n + 3) + 3 * layout.len_width + layout.shift_width + layout.step_aux),
        "range": [1, int(layout.T_max)],
        "ops": {"x": 0, "cx": 0, "ccx": 0, "h": 0, "measure": 0, "reset": 0},
    }
    with tempfile.TemporaryDirectory() as td:
        eea_json = Path(td) / "synthetic_eea_counts.json"
        eea_json.write_text(json.dumps(synthetic_eea, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        args = SimpleNamespace(
            n=small_n,
            p=small_p,
            s_qubits=None,
            point_constant="zero",
            x2=None,
            y2=None,
            eea_steps_json=str(eea_json),
            allow_eea_n_mismatch=False,
            mcx_policy="clean-vchain",
            validate_full_mul=True,
            max_full_recursive_n=max(8, small_n),
        )
        report = build_report(args)

    width = report["qiskit_width_report"]
    _assert(width["num_qubits"] == 1 + 3 * small_n + shared_eea_s_qubits(small_n), f"counter width report mismatch: {width}")
    _assert(width["has_only_ctrl_X_Y_A_S_qregs"] is True, f"unexpected qreg layout in counter width report: {width}")
    _assert(width["has_extra_E_or_R_registers"] is False, f"counter should not report extra E/R registers: {width}")
    _assert(report["eea_meta"]["skipped_alg3_steps_in_wrapper"] == layout.T_max, f"EEA wrapper did not stop on T_max step instructions: {report['eea_meta']}")
    _assert(report["validation"] and report["validation"].get("all_passed"), f"report validation failed: {report.get('validation')}")
    _assert(report["key_ccx"]["point_addition_fig14_total"] > 0, f"total point-addition CCX count should be positive: {report['key_ccx']}")
    for name, summary in report["block_summaries"].items():
        opaque_terms = summary.get("opaque_terms", {})
        if opaque_terms:
            raise AssertionError(f"block summary {name!r} contains opaque terms: {opaque_terms}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict tests for the S835 fast-dual wrapped point-addition implementation")
    parser.add_argument("--n", type=int, default=4, help="small width used for recursive compiled-block validation")
    parser.add_argument("--p", type=lambda s: int(s, 0), default=13, help="small prime modulus used with --n")
    parser.add_argument("--skip-n256", action="store_true", help="skip the lightweight n=256 S835 width construction check")
    parser.add_argument("--skip-report", action="store_true", help="skip the tiny integrated point-addition counter report test")
    parser.add_argument("--verbose", action="store_true", help="print individual test timing")
    args = parser.parse_args()

    tests: list[TestItem] = [
        TestItem("S835 width/register layout", lambda: test_s835_width_and_register_layout(small_n=args.n, small_p=args.p, skip_n256=args.skip_n256)),
        TestItem("Fig.14/Fig.15 top-level point-addition schedule", lambda: test_fig14_top_level_schedule(small_n=args.n, small_p=args.p)),
        TestItem("compiled arithmetic block assembly", lambda: test_compiled_arithmetic_block_assembly(small_n=args.n, small_p=args.p)),
    ]
    if not args.skip_report:
        tests.append(TestItem("tiny integrated point-addition counter report", lambda: test_tiny_point_addition_counter_report(small_n=args.n, small_p=args.p)))

    print("Strict S835 wrapped point-addition tests")
    print(f"small test parameters: n={args.n}, p={args.p}\n")

    failed: list[tuple[str, str]] = []
    t_all = time.perf_counter()
    for item in tests:
        t0 = time.perf_counter()
        try:
            item.fn()
            print(f"[PASS] {item.name} ({time.perf_counter() - t0:.2f}s)")
        except Exception as exc:
            print(f"[FAIL] {item.name} ({time.perf_counter() - t0:.2f}s)")
            print(f"       {type(exc).__name__}: {exc}")
            failed.append((item.name, f"{type(exc).__name__}: {exc}"))
        if args.verbose:
            sys.stdout.flush()

    print(f"\nSummary: {len(tests) - len(failed)}/{len(tests)} test groups passed in {time.perf_counter() - t_all:.2f}s")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
