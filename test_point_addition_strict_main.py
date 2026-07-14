import argparse
import gc
import json
import math
import multiprocessing
import random
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterable, Sequence


def _require_qiskit_or_skip() -> None:
    try:
        import qiskit
    except Exception as exc:
        print(f"[SKIP] Qiskit is not installed, so point-addition circuit tests were skipped: {exc}")
        raise SystemExit(0)


_require_qiskit_or_skip()

import qiskit
import eea_circuit_s835_fastdual as eea
from run_eea_s835_fastdual_recursive_chunks_checkpoint import count_range
from test_eea_strict_main import classical_expected_for_algorithm3
from ccx_recursive_block_counter import CounterPolicy
from count_s835_fastdual_wrapped_point_addition_blocks_compiled import (
    build_report,
    count_compiled_arithmetic_subblocks,
    validate_full_mul_square,
    assemble_mul_square_from_compiled_primitives,
)
from point_addition_fig14_s835_fastdual_wrapped_quadratic import (
    SECP256K1_P,
    build_point_addition_fig14_quadratic,
)
from under1000_eea_shared_s835_fastdual_wrapped import (
    shared_eea_layout,
    shared_eea_s_qubits,
)


@dataclass(frozen=True)
class TestItem:
    name: str
    fn: Callable[[], object]


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


def _clear_project_caches() -> None:
    """Release cached Qiskit gate definitions between large-width cases."""
    clear_backend = getattr(eea, "clear_gate_construction_caches", None)
    if callable(clear_backend):
        clear_backend()

    prefixes = (
        "eea_circuit_",
        "quadratic_",
        "under1000_",
        "ccx_recursive_block_counter",
        "count_s835_fastdual_",
    )
    seen: set[int] = set()
    for module_name, module in list(sys.modules.items()):
        if module is None or not module_name.startswith(prefixes):
            continue
        for obj in vars(module).values():
            oid = id(obj)
            if oid in seen:
                continue
            seen.add(oid)
            clear = getattr(obj, "cache_clear", None)
            if callable(clear):
                clear()
    gc.collect()


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



P_12 = 4093
P_16 = 65521
P_20 = 1048573
P_24 = 16777213
P_28 = 268435399
P_31 = 2147483647
P_32 = 4294967291
P_40 = 1099511627689
P_48 = 281474976710597
P_56 = 72057594037927931
P_61 = 2305843009213693951
P_64 = 18446744073709551557
P_96 = 79228162514264337593543950319
P_128 = 340282366762482138434845932244680310783
P_160 = 1461501637330902918203684832716283019653785059327
P_192 = 6277101735386680763835789423207666416083908700390324961279
P_224 = 26959946667150639794667015087019630673557916260026308143510066298881
P_384 = 39402006196394479212279040100143613805079739270465446667948293404245721771496870329047266088258938001861606973112319
P_512 = 13407807929942597099574024998205846127479365820592393377723561443721764030073546976801874298166903427690031858186486050853753882811946569946433649006083527

# The complete default matrix spans the widths used by the numerical resource
# estimates (64--512 bits) and also retains smaller widths that exercise the
# circuit builders quickly.  All values are fixed prime constants.  P_128,
# P_160, P_192, P_224, SECP256K1_P, and P_384 are standard prime-field curve
# moduli; P_96 and P_512 are fixed primes immediately below powers of two.
DEFAULT_LARGE_PRIMES = (
    P_12,
    P_16,
    P_20,
    P_24,
    P_28,
    P_31,
    P_32,
    P_40,
    P_48,
    P_56,
    P_61,
    P_64,
    P_96,
    P_128,
    P_160,
    P_192,
    P_224,
    SECP256K1_P,
    P_384,
    P_512,
)

# Full compiled-arithmetic assembly is more memory intensive than layout,
# schedule, and recursive step counting.  It is therefore run in isolated
# worker processes for a representative subset through 64 bits.  This keeps the
# default release suite bounded while still exercising much larger widths via
# the lightweight production-circuit paths above.
DEFAULT_COMPILED_PRIMES = (
    P_12,
    P_16,
    P_20,
    P_31,
    P_32,
    P_40,
    P_48,
    P_61,
    P_64,
)

_KNOWN_LARGE_PRIMES = frozenset(DEFAULT_LARGE_PRIMES)
_MR_BASES_64 = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)
_MR_BASES_LARGE = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53)


@dataclass(frozen=True)
class PrimeVector:
    p: int
    n: int
    x: int
    source: str
    inverse: int
    exact_steps: int
    iterations: int
    max_quotient_bits: int


@dataclass(frozen=True)
class ClassicalProfile:
    iterations: int
    max_quotient_bits: int


def is_prime(n: int) -> bool:
    """Validate configured primes and screen custom moduli.

    The Miller--Rabin basis is deterministic for ``n < 2**64``.  The fixed
    larger constants in the default matrix are explicitly whitelisted after
    independent primality verification.  For a user-supplied value above
    64 bits, the function performs a strong probable-prime screen using a
    conservative collection of small bases; this is a test preflight, not a
    general-purpose primality certificate.
    """
    if n < 2:
        return False
    if n in _KNOWN_LARGE_PRIMES:
        return True
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    bases = _MR_BASES_64 if n < (1 << 64) else _MR_BASES_LARGE
    for a in bases:
        if a % n == 0:
            continue
        y = pow(a, d, n)
        if y in (1, n - 1):
            continue
        for _ in range(s - 1):
            y = (y * y) % n
            if y == n - 1:
                break
        else:
            return False
    return True


def classical_profile(p: int, x: int) -> ClassicalProfile:
    x_used = min(x, p - x)
    r_prev, r = p, x_used
    iterations = 0
    max_quotient_bits = 0
    while r:
        q = r_prev // r
        max_quotient_bits = max(max_quotient_bits, q.bit_length())
        iterations += 1
        r_prev, r = r, r_prev - q * r
    return ClassicalProfile(iterations=iterations, max_quotient_bits=max_quotient_bits)


def fibonacci_stress_x(p: int) -> int:
    """Return a Fibonacci-like input below p/2 to induce many EEA iterations."""
    a, b = 1, 1
    limit = p // 2
    while a + b < limit:
        a, b = b, a + b
    return max(1, b)


def representative_inputs(p: int, *, seed: int, random_count: int) -> list[tuple[int, str]]:
    items: dict[int, str] = {}

    def add(x: int, source: str) -> None:
        if 1 <= x < p and math.gcd(x, p) == 1:
            items.setdefault(x, source)

    for x, source in (
        (1, "edge:one"),
        (2, "edge:two"),
        (3, "edge:three"),
        (p // 2, "edge:half-low"),
        (p // 2 + 1, "edge:half-high"),
        (p - 3, "edge:p-3"),
        (p - 2, "edge:p-2"),
        (p - 1, "edge:p-1"),
    ):
        add(x, source)

    fib_x = fibonacci_stress_x(p)
    add(fib_x, "stress:fibonacci")
    add(p - fib_x, "stress:fibonacci-symmetric")

    rng = random.Random((seed << 32) ^ p)
    added = 0
    attempts = 0
    while added < random_count and attempts < 1000:
        attempts += 1
        x = rng.randrange(1, p)
        before = len(items)
        add(x, f"random:{seed}:{added + 1}")
        if len(items) > before:
            added += 1

    return sorted(items.items())


def test_classical_large_prime_oracle(
    primes: Sequence[int], *, seed: int, random_count: int
) -> dict:
    vectors: list[PrimeVector] = []
    for p in primes:
        if not is_prime(p):
            raise AssertionError(f"configured test modulus p={p} is not prime")
        n = p.bit_length()
        for x, source in representative_inputs(p, seed=seed, random_count=random_count):
            expected = classical_expected_for_algorithm3(p, x)
            profile = classical_profile(p, x)
            inverse = pow(x, -1, p)

            if expected.inverse != inverse:
                raise AssertionError(
                    f"classical oracle mismatch p={p}, x={x}: "
                    f"got {expected.inverse}, expected {inverse}"
                )
            if (x * inverse) % p != 1:
                raise AssertionError(f"inverse identity failed p={p}, x={x}")
            if expected.x_used != min(x, p - x):
                raise AssertionError(f"preprocessing mismatch p={p}, x={x}")
            reconstructed = (
                expected.tprime_final
                if expected.iter_final
                else (-expected.tprime_final) % p
            )
            if reconstructed != inverse:
                raise AssertionError(
                    f"t'/Iter reconstruction failed p={p}, x={x}: "
                    f"got {reconstructed}, expected {inverse}"
                )
            if expected.exact_step_count <= 0 or expected.exact_step_count % 4:
                raise AssertionError(
                    f"invalid Algorithm-3 step profile p={p}, x={x}: "
                    f"{expected.exact_step_count}"
                )
            if x not in (p - x,):
                sym = pow(p - x, -1, p)
                if sym != (-inverse) % p:
                    raise AssertionError(
                        f"inverse symmetry failed p={p}, x={x}: "
                        f"inv(p-x)={sym}, expected {(-inverse) % p}"
                    )

            vectors.append(
                PrimeVector(
                    p=p,
                    n=n,
                    x=x,
                    source=source,
                    inverse=inverse,
                    exact_steps=expected.exact_step_count,
                    iterations=profile.iterations,
                    max_quotient_bits=profile.max_quotient_bits,
                )
            )

    return {
        "prime_count": len(primes),
        "vector_count": len(vectors),
        "vectors": [asdict(v) for v in vectors],
    }


def expected_step_width(n: int, *, len_width: int, shift_width: int, aux_size: int) -> int:
    # Phase1, Phase2, Iter, Sign; Work1/Work2; l_t/l_q/l_rp; l_s; Aux.
    return 4 + 2 * (n + 3) + 3 * len_width + shift_width + aux_size


def sampled_steps(T_max: int) -> list[int]:
    return sorted(set((1, 2, 3, 4, max(1, T_max // 2), T_max)))


def test_fastdual_large_width_build_and_count(primes: Sequence[int]) -> dict:
    records: list[dict] = []
    for n in sorted({p.bit_length() for p in primes}):
        cfg = eea.get_n_config(n)
        len_width = int(cfg["len_width"])
        shift_width = int(cfg["shift_width"])
        T_max = int(cfg["T_max"])
        aux_size = int(eea.qiskit_paper_aux_size(n, len_width, shift_width, T_max))
        expected_width = expected_step_width(
            n, len_width=len_width, shift_width=shift_width, aux_size=aux_size
        )

        for T in sampled_steps(T_max):
            windows = eea.active_windows(n, T)
            invalid = {name: interval for name, interval in windows.items() if interval[0] > interval[1]}
            if invalid:
                raise AssertionError(f"invalid active window n={n}, T={T}: {invalid}")

            qc = eea.build_step_circuit(
                n,
                T,
                T_max=T_max,
                aux_size=aux_size,
                measurement_uncompute=False,
            )
            if qc.num_qubits != expected_width:
                raise AssertionError(
                    f"step width mismatch n={n}, T={T}: "
                    f"got {qc.num_qubits}, expected {expected_width}"
                )
            ops = Counter(eea.count_circuit_ops_recursive(qc))
            unexpected = set(ops) - {"x", "cx", "ccx"}
            if unexpected:
                raise AssertionError(
                    f"unexpected reversible leaves n={n}, T={T}: {sorted(unexpected)}"
                )
            if ops["ccx"] <= 0 or sum(ops.values()) <= 0:
                raise AssertionError(f"empty recursive count n={n}, T={T}: {dict(ops)}")
            records.append(
                {
                    "n": n,
                    "T": T,
                    "T_max": T_max,
                    "measurement_uncompute": False,
                    "num_qubits": qc.num_qubits,
                    "top_level_instructions": len(qc.data),
                    "ops": dict(sorted(ops.items())),
                }
            )

        # Check the dynamic measurement-uncompute form on an end-of-iteration step.
        T_mb = 4 if T_max >= 4 else T_max
        qc_mb = eea.build_step_circuit(
            n,
            T_mb,
            T_max=T_max,
            aux_size=aux_size,
            measurement_uncompute=True,
        )
        ops_mb = Counter(eea.count_circuit_ops_recursive(qc_mb))
        unexpected_mb = set(ops_mb) - {"x", "cx", "ccx", "h", "z", "cz", "measure", "reset"}
        if unexpected_mb:
            raise AssertionError(
                f"unexpected measurement-mode leaves n={n}, T={T_mb}: "
                f"{sorted(unexpected_mb)}"
            )
        if ops_mb["measure"] <= 0 or ops_mb["reset"] <= 0:
            raise AssertionError(
                f"measurement-uncompute path missing measure/reset n={n}, T={T_mb}: "
                f"{dict(ops_mb)}"
            )
        records.append(
            {
                "n": n,
                "T": T_mb,
                "T_max": T_max,
                "measurement_uncompute": True,
                "num_qubits": qc_mb.num_qubits,
                "top_level_instructions": len(qc_mb.data),
                "ops": dict(sorted(ops_mb.items())),
            }
        )
        _clear_project_caches()

    return {"width_count": len({p.bit_length() for p in primes}), "records": records}


def test_checkpoint_chunk_aggregation(*, n: int = 16, start: int = 1, end: int = 4) -> dict:
    cfg = eea.get_n_config(n)
    T_max = int(cfg["T_max"])
    len_width = int(cfg["len_width"])
    shift_width = int(cfg["shift_width"])
    aux_size = int(eea.qiskit_paper_aux_size(n, len_width, shift_width, T_max))

    chunk = count_range(
        n,
        T_max,
        start,
        end,
        aux_size=aux_size,
        measurement_uncompute=True,
    )
    expected = Counter()
    for T in range(start, end + 1):
        qc = eea.build_step_circuit(
            n,
            T,
            T_max=T_max,
            aux_size=aux_size,
            measurement_uncompute=True,
        )
        expected += Counter(eea.count_circuit_ops_recursive(qc))

    got = Counter({str(k): int(v) for k, v in chunk["ops"].items()})
    if got != expected:
        raise AssertionError(
            "checkpoint chunk aggregation mismatch:\n"
            f"got={dict(sorted(got.items()))}\n"
            f"expected={dict(sorted(expected.items()))}"
        )
    if chunk["range"] != [start, end] or int(chunk["n"]) != n:
        raise AssertionError(f"unexpected chunk metadata: {chunk}")

    return {
        "n": n,
        "range": [start, end],
        "T_max": T_max,
        "num_qubits": int(chunk["num_qubits"]),
        "ops": dict(sorted(got.items())),
    }


def _format_prime(p: int) -> str:
    if p.bit_length() <= 64:
        return str(p)
    text = f"{p:x}"
    return f"0x{text[:12]}...{text[-12:]}"


def test_large_prime_point_addition_layout_schedule_matrix(
    primes: Sequence[int],
) -> dict:
    """Exercise p-specific wrapped point-addition construction at every width.

    This path is intentionally lightweight: it checks the exact S835 register
    layout and the Fig.14/Fig.15 dynamic schedule without recursively compiling
    every arithmetic primitive.  The latter is covered separately on a selected
    subset in isolated worker processes.
    """
    records: list[dict] = []
    for p in primes:
        n = p.bit_length()
        t0 = time.perf_counter()
        test_s835_width_and_register_layout(small_n=n, small_p=p, skip_n256=True)
        test_fig14_top_level_schedule(small_n=n, small_p=p)
        records.append({"p": p, "n": n, "elapsed_s": time.perf_counter() - t0})
        _clear_project_caches()
    return {
        "prime_count": len(primes),
        "width_count": len({p.bit_length() for p in primes}),
        "records": records,
    }


_WORKER_RESULT_PREFIX = "__POINT_ADDITION_COMPILED_WORKER__="


def _run_compiled_point_addition_worker(p: int) -> dict:
    """Run one memory-intensive compiled-arithmetic case in this process."""
    n = p.bit_length()
    t0 = time.perf_counter()
    test_s835_width_and_register_layout(small_n=n, small_p=p, skip_n256=True)
    test_fig14_top_level_schedule(small_n=n, small_p=p)
    test_compiled_arithmetic_block_assembly(small_n=n, small_p=p)
    return {"p": p, "n": n, "elapsed_s": time.perf_counter() - t0}


def _compiled_worker_pipe_entry(p: int, send_conn) -> None:
    """Fork-worker entry point; report either a result or a traceback."""
    try:
        detail = _run_compiled_point_addition_worker(p)
        send_conn.send({"status": "PASS", "detail": detail})
    except BaseException:  # pragma: no cover - exercised only on worker failure
        send_conn.send({"status": "FAIL", "error": traceback.format_exc()})
    finally:
        send_conn.close()


def _run_compiled_case_fork(p: int, *, timeout_s: float) -> dict:
    ctx = multiprocessing.get_context("fork")
    recv_conn, send_conn = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=_compiled_worker_pipe_entry,
        args=(p, send_conn),
        name=f"point-addition-compiled-n{p.bit_length()}",
    )
    process.start()
    send_conn.close()
    process.join(timeout_s)
    if process.is_alive():
        process.terminate()
        process.join(5.0)
        if process.is_alive():
            process.kill()
            process.join()
        recv_conn.close()
        raise AssertionError(
            f"compiled point-addition fork worker timed out for p={p} "
            f"(n={p.bit_length()}) after {timeout_s:g}s"
        )

    message = recv_conn.recv() if recv_conn.poll() else None
    recv_conn.close()
    exitcode = process.exitcode
    process.close()
    if exitcode != 0:
        raise AssertionError(
            f"compiled point-addition fork worker exited with code {exitcode} "
            f"for p={p} (n={p.bit_length()})"
        )
    if not message:
        raise AssertionError(
            f"compiled point-addition fork worker returned no payload for p={p}"
        )
    if message.get("status") != "PASS":
        raise AssertionError(
            f"compiled point-addition fork worker failed for p={p} "
            f"(n={p.bit_length()}):\n{message.get('error', 'unknown error')}"
        )
    return dict(message["detail"])


def _run_compiled_case_subprocess(
    p: int, *, timeout_s: float, script: Path
) -> dict:
    command = [
        sys.executable,
        str(script),
        "--_compiled-worker-prime",
        str(p),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=script.parent,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(
            f"compiled point-addition worker timed out for p={p} "
            f"(n={p.bit_length()}) after {timeout_s:g}s"
        ) from exc

    if completed.returncode != 0:
        stdout = completed.stdout[-4000:]
        stderr = completed.stderr[-4000:]
        raise AssertionError(
            f"compiled point-addition worker failed for p={p} "
            f"(n={p.bit_length()}), exit={completed.returncode}\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )

    payload = None
    for line in completed.stdout.splitlines():
        if line.startswith(_WORKER_RESULT_PREFIX):
            payload = json.loads(line[len(_WORKER_RESULT_PREFIX):])
    if payload is None:
        raise AssertionError(
            f"compiled point-addition worker produced no result marker for "
            f"p={p}; stdout={completed.stdout[-4000:]!r}"
        )
    return payload


def test_large_prime_compiled_arithmetic_matrix(
    primes: Sequence[int], *, timeout_s: float, verbose: bool = False
) -> dict:
    """Compile selected p-specific arithmetic blocks in isolated workers.

    Qiskit circuit definitions and transpilation caches can retain substantial
    memory when many widths are compiled in one interpreter.  One process per
    modulus makes the release test deterministic and releases all associated
    memory after each case.  POSIX systems use ``fork`` to avoid duplicating the
    parent interpreter's imported Qiskit state; other platforms fall back to a
    fresh Python subprocess.
    """
    script = Path(__file__).resolve()
    use_fork = "fork" in multiprocessing.get_all_start_methods()
    worker_mode = "fork" if use_fork else "subprocess"
    records: list[dict] = []
    _clear_project_caches()
    for index, p in enumerate(primes, start=1):
        if verbose:
            print(
                f"       compiled worker {index}/{len(primes)}: "
                f"n={p.bit_length()}, p={_format_prime(p)}",
                flush=True,
            )
        t0 = time.perf_counter()
        payload = (
            _run_compiled_case_fork(p, timeout_s=timeout_s)
            if use_fork
            else _run_compiled_case_subprocess(
                p, timeout_s=timeout_s, script=script
            )
        )
        payload["wall_elapsed_s"] = time.perf_counter() - t0
        payload["worker_mode"] = worker_mode
        records.append(payload)
        _clear_project_caches()
        if verbose:
            print(
                f"         passed in {payload['wall_elapsed_s']:.2f}s "
                f"(worker {payload['elapsed_s']:.2f}s)",
                flush=True,
            )

    return {
        "prime_count": len(primes),
        "max_n": max((p.bit_length() for p in primes), default=0),
        "isolated_workers": True,
        "worker_mode": worker_mode,
        "worker_timeout_s": timeout_s,
        "records": records,
    }

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Strict wrapped point-addition tests with integrated larger-prime "
            "construction/resource regressions"
        )
    )
    parser.add_argument(
        "--n",
        type=int,
        default=4,
        help="small width used for the fast compiled-block validation",
    )
    parser.add_argument(
        "--p",
        type=lambda s: int(s, 0),
        default=13,
        help="small prime modulus used with --n",
    )
    parser.add_argument(
        "--skip-n256",
        action="store_true",
        help="skip the lightweight n=256 S835 width construction check",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="skip the tiny integrated point-addition counter report test",
    )
    parser.add_argument(
        "--skip-large-primes",
        action="store_true",
        help="run only the original small point-addition checks",
    )
    parser.add_argument(
        "--large-primes",
        "--primes",
        dest="large_primes",
        type=lambda s: int(s, 0),
        nargs="*",
        default=None,
        help=(
            "prime matrix used by the classical oracle, large-width step "
            "counter, and point-addition layout/schedule checks; the default "
            "contains 20 widths from 12 through 512 bits"
        ),
    )
    parser.add_argument(
        "--compiled-primes",
        type=lambda s: int(s, 0),
        nargs="*",
        default=None,
        help=(
            "subset receiving the heavier compiled-arithmetic point-addition "
            "check; by default a representative 12--64-bit subset is used. "
            "When --large-primes is explicitly supplied and this option is "
            "omitted, the custom large-prime list is also compiled."
        ),
    )
    parser.add_argument(
        "--compiled-worker-timeout-s",
        type=float,
        default=300.0,
        help="per-modulus timeout for isolated compiled-arithmetic workers",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=835,
        help="deterministic seed for generated large-prime x values",
    )
    parser.add_argument(
        "--random-x",
        type=int,
        default=2,
        help="number of deterministic random x values per larger prime",
    )
    parser.add_argument(
        "--skip-large-chunk",
        "--skip-chunk",
        dest="skip_large_chunk",
        action="store_true",
        help="skip the n=16 checkpoint/chunk aggregation regression",
    )
    parser.add_argument(
        "--skip-large-point-addition",
        "--skip-point-addition",
        dest="skip_large_point_addition",
        action="store_true",
        help="skip all p-specific point-addition checks for the larger-prime matrix",
    )
    parser.add_argument(
        "--skip-large-compiled",
        action="store_true",
        help="keep large-width layout/schedule checks but skip compiled arithmetic workers",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="optional JSON report path",
    )
    parser.add_argument("--verbose", action="store_true", help="print additional test details")
    parser.add_argument(
        "--_compiled-worker-prime",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args._compiled_worker_prime is not None:
        worker_p = int(args._compiled_worker_prime)
        if not is_prime(worker_p):
            raise SystemExit(f"compiled worker modulus p={worker_p} is not prime")
        detail = _run_compiled_point_addition_worker(worker_p)
        print(_WORKER_RESULT_PREFIX + json.dumps(detail, sort_keys=True))
        return

    custom_large_matrix = args.large_primes is not None
    large_primes = list(
        dict.fromkeys(
            DEFAULT_LARGE_PRIMES if args.large_primes is None else args.large_primes
        )
    )
    if args.compiled_primes is not None:
        compiled_primes = list(dict.fromkeys(args.compiled_primes))
    elif custom_large_matrix:
        # Preserve the previous command-line behavior: a custom --large-primes
        # list receives the complete point-addition block check unless the user
        # explicitly chooses a smaller --compiled-primes subset.
        compiled_primes = list(large_primes)
    else:
        compiled_primes = list(DEFAULT_COMPILED_PRIMES)

    # A compiled modulus should also be represented in the oracle, production
    # step, and layout/schedule matrices.
    large_primes = list(dict.fromkeys([*large_primes, *compiled_primes]))
    if not args.skip_large_primes and not large_primes:
        raise SystemExit("at least one larger prime is required unless --skip-large-primes is used")
    if args.compiled_worker_timeout_s <= 0:
        raise SystemExit("--compiled-worker-timeout-s must be positive")

    tests: list[TestItem] = [
        TestItem(
            "S835 width/register layout",
            lambda: test_s835_width_and_register_layout(
                small_n=args.n,
                small_p=args.p,
                skip_n256=args.skip_n256,
            ),
        ),
        TestItem(
            "Fig.14/Fig.15 top-level point-addition schedule",
            lambda: test_fig14_top_level_schedule(small_n=args.n, small_p=args.p),
        ),
        TestItem(
            "compiled arithmetic block assembly",
            lambda: test_compiled_arithmetic_block_assembly(
                small_n=args.n,
                small_p=args.p,
            ),
        ),
    ]
    if not args.skip_report:
        tests.append(
            TestItem(
                "tiny integrated point-addition counter report",
                lambda: test_tiny_point_addition_counter_report(
                    small_n=args.n,
                    small_p=args.p,
                ),
            )
        )

    if not args.skip_large_primes:
        tests.append(
            TestItem(
                "large-prime classical EEA oracle matrix",
                lambda: test_classical_large_prime_oracle(
                    large_primes,
                    seed=args.seed,
                    random_count=max(0, args.random_x),
                ),
            )
        )
        tests.append(
            TestItem(
                "large-width S835 fast-dual step construction/counting",
                lambda: test_fastdual_large_width_build_and_count(large_primes),
            )
        )
        if not args.skip_large_chunk:
            tests.append(
                TestItem(
                    "checkpoint chunk aggregation at n=16",
                    lambda: test_checkpoint_chunk_aggregation(n=16, start=1, end=4),
                )
            )
        if not args.skip_large_point_addition:
            tests.append(
                TestItem(
                    "large-prime wrapped point-addition layout/schedule matrix",
                    lambda: test_large_prime_point_addition_layout_schedule_matrix(
                        large_primes
                    ),
                )
            )
        # Keep the isolated compiled workers last.  Each worker releases its
        # process-local Qiskit definitions on exit, and no subsequent heavy
        # in-process circuit construction is required.
        if (
            not args.skip_large_point_addition
            and not args.skip_large_compiled
            and compiled_primes
        ):
            tests.append(
                TestItem(
                    "selected large-prime compiled arithmetic matrix",
                    lambda: test_large_prime_compiled_arithmetic_matrix(
                        compiled_primes,
                        timeout_s=args.compiled_worker_timeout_s,
                        verbose=args.verbose,
                    ),
                )
            )

    print("Strict S835 wrapped point-addition tests")
    print(f"Qiskit: {qiskit.__version__}")
    print(f"small test parameters: n={args.n}, p={args.p}")
    if args.skip_large_primes:
        print("larger-prime regressions: skipped\n")
    else:
        widths = sorted({p.bit_length() for p in large_primes})
        compiled_widths = sorted({p.bit_length() for p in compiled_primes})
        print(f"larger-prime matrix: {len(large_primes)} primes / {len(widths)} widths")
        print("large-width n values: " + ", ".join(map(str, widths)))
        if args.skip_large_point_addition or args.skip_large_compiled:
            print("compiled-arithmetic large-prime matrix: skipped")
        else:
            print(
                f"compiled-arithmetic subset: {len(compiled_primes)} primes; "
                "n=" + ", ".join(map(str, compiled_widths))
            )
        if args.verbose:
            compiled_active = not (
                args.skip_large_point_addition or args.skip_large_compiled
            )
            for p in large_primes:
                mode = (
                    "compiled"
                    if compiled_active and p in compiled_primes
                    else "layout/count"
                )
                print(
                    f"  n={p.bit_length():3d}  p={_format_prime(p):>29s}  "
                    f"scope={mode}"
                )
        print(
            "large-p scope: classical oracle plus construction/resource regressions; "
            "not an exhaustive endpoint proof.\n"
        )

    results: dict[str, dict] = {}
    failures: list[tuple[str, str]] = []
    t_all = time.perf_counter()
    for item in tests:
        t0 = time.perf_counter()
        try:
            detail = item.fn()
            elapsed = time.perf_counter() - t0
            results[item.name] = {
                "status": "PASS",
                "elapsed_s": elapsed,
                "detail": detail if detail is not None else {},
            }
            print(f"[PASS] {item.name} ({elapsed:.2f}s)")
            if args.verbose and isinstance(detail, dict):
                if "vector_count" in detail:
                    print(f"       vectors={detail['vector_count']}")
                elif "records" in detail:
                    print(f"       records={len(detail['records'])}")
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            message = f"{type(exc).__name__}: {exc}"
            results[item.name] = {
                "status": "FAIL",
                "elapsed_s": elapsed,
                "error": message,
            }
            failures.append((item.name, message))
            print(f"[FAIL] {item.name} ({elapsed:.2f}s)")
            print(f"       {message}")
        if args.verbose:
            sys.stdout.flush()

    elapsed_all = time.perf_counter() - t_all
    print(
        f"\nSummary: {len(tests) - len(failures)}/{len(tests)} "
        f"test groups passed in {elapsed_all:.2f}s"
    )

    report = {
        "test": "point-addition-strict-with-integrated-large-prime-regressions",
        "qiskit_version": qiskit.__version__,
        "small_parameters": {"n": args.n, "p": args.p},
        "large_primes": [] if args.skip_large_primes else large_primes,
        "compiled_primes": (
            []
            if args.skip_large_primes
            or args.skip_large_point_addition
            or args.skip_large_compiled
            else compiled_primes
        ),
        "large_widths": (
            [] if args.skip_large_primes else sorted({p.bit_length() for p in large_primes})
        ),
        "large_prime_scope": (
            "Classical large-prime EEA oracle validation and large-width circuit "
            "construction/resource regressions through 512 bits. Point-addition "
            "layout/schedule checks cover the full matrix; selected moduli through "
            "64 bits additionally receive isolated compiled-arithmetic checks. "
            "This is not exhaustive end-to-end semantic verification of the "
            "optimized EEA endpoint."
        ),
        "summary": {
            "groups": len(tests),
            "passed": len(tests) - len(failures),
            "failed": len(failures),
            "elapsed_s": elapsed_all,
        },
        "results": results,
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Report: {args.report}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
