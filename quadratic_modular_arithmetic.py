"""Quadratic double-and-add modular arithmetic backend.

This replaces the previous MCX/lexicographic fallback with linear-Toffoli
subblocks:

* quantum-quantum controlled add/sub uses a Cuccaro/TTK style linear controlled
  adder plus a linear Cuccaro high-bit comparator;
* all constant corrections/comparisons use Gidney-style measurement-vented
  classical-quantum adders/comparators with borrowed dirty bits;
* modular multiplication and squaring keep the RNSL/PDF dbl/add schedule.

The resulting multiplier has n controlled modular additions and n-1 modular
Doublings, each O(n) Toffoli/CNOT, so the multiplier is O(n^2) Toffoli/CNOT.
"""
from functools import lru_cache
from typing import Optional, Sequence

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Clbit, Instruction, Qubit

from quadratic_lazy_instruction import LazyDefinedInstruction
from quadratic_gidney_arithmetic import (
    append_controlled_add_mod2_with_carry_flag,
    append_controlled_sub_mod2_with_carry_flag,
    append_gidney_add_const_mod2n,
    append_gidney_sub_const_mod2n,
    append_gidney_compare_ge_const,
    append_gidney_compare_lt_const,
    append_xor_lt_quantum_cuccaro,
)

SECP256K1_P = (1 << 256) - (1 << 32) - 977


def _as_list(xs: Sequence[Qubit]) -> list[Qubit]:
    return list(xs)


def _bit_mask(n: int) -> int:
    return (1 << n) - 1


def _swap_xor(qc: QuantumCircuit, a: Qubit, b: Qubit) -> None:
    qc.cx(a, b); qc.cx(b, a); qc.cx(a, b)


def append_cyclic_left_shift(qc: QuantumCircuit, reg: Sequence[Qubit]) -> None:
    r = _as_list(reg)
    for i in reversed(range(len(r) - 1)):
        _swap_xor(qc, r[i], r[i + 1])


def append_cyclic_right_shift(qc: QuantumCircuit, reg: Sequence[Qubit]) -> None:
    r = _as_list(reg)
    for i in range(len(r) - 1):
        _swap_xor(qc, r[i], r[i + 1])


def _need_cbits(cbits: Sequence[Clbit], n: int) -> list[Clbit]:
    cb = list(cbits)
    if len(cb) < max(1, n):
        raise ValueError(f"need at least {max(1,n)} classical work bits")
    return cb


def append_ctrl_add_modp_quadratic(qc: QuantumCircuit, ctrl: Qubit, addend: Sequence[Qubit], acc: Sequence[Qubit], aux5: Sequence[Qubit], cbits: Sequence[Clbit], *, p: int = SECP256K1_P) -> None:
    addend = _as_list(addend); acc = _as_list(acc); aux5 = _as_list(aux5); n = len(acc)
    if len(addend) != n or len(aux5) < 5:
        raise ValueError("bad ctrl_add_modp widths")
    cb = _need_cbits(cbits, n)
    carry, flag = aux5[0], aux5[1]
    clean3 = [aux5[0], aux5[2], aux5[3]]
    # 1. acc += ctrl*addend mod 2^n, flag ^= carry_out.
    append_controlled_add_mod2_with_carry_flag(qc, ctrl, addend, acc, carry, flag)
    # 2. flag ^= [acc >= p].  Since inputs are canonical, ctrl=0 implies acc<p.
    append_gidney_compare_ge_const(qc, acc, p, flag, addend, clean3, cb)
    # 3. if flag: acc -= p.  Borrow addend as dirty.
    append_gidney_sub_const_mod2n(qc, acc, p, addend[: max(0, n - 1)], clean3, cb, ctrl=flag)
    # 4. Uncompute flag using final acc < addend.
    append_xor_lt_quantum_cuccaro(qc, acc, addend, flag, carry, ctrl=ctrl)


def append_ctrl_sub_modp_quadratic(qc: QuantumCircuit, ctrl: Qubit, subtrahend: Sequence[Qubit], acc: Sequence[Qubit], aux5: Sequence[Qubit], cbits: Sequence[Clbit], *, p: int = SECP256K1_P) -> None:
    # Hand inverse of append_ctrl_add_modp_quadratic.
    subtrahend = _as_list(subtrahend); acc = _as_list(acc); aux5 = _as_list(aux5); n = len(acc)
    cb = _need_cbits(cbits, n)
    carry, flag = aux5[0], aux5[1]
    clean3 = [aux5[0], aux5[2], aux5[3]]
    append_xor_lt_quantum_cuccaro(qc, acc, subtrahend, flag, carry, ctrl=ctrl)
    append_gidney_add_const_mod2n(qc, acc, p, subtrahend[: max(0, n - 1)], clean3, cb, ctrl=flag)
    append_gidney_compare_ge_const(qc, acc, p, flag, subtrahend, clean3, cb)
    append_controlled_sub_mod2_with_carry_flag(qc, ctrl, subtrahend, acc, carry, flag)


def append_add_const_modp_quadratic(qc: QuantumCircuit, target: Sequence[Qubit], const: int, dirty: Sequence[Qubit], aux4: Sequence[Qubit], cbits: Sequence[Clbit], *, p: int = SECP256K1_P, ctrl: Optional[Qubit] = None) -> None:
    target = _as_list(target); dirty = _as_list(dirty); aux4 = _as_list(aux4); n = len(target)
    if len(dirty) < n or len(aux4) < 4:
        raise ValueError("bad const modular add widths")
    cb = _need_cbits(cbits, n)
    flag = aux4[0]
    clean3 = [aux4[1], aux4[2], aux4[3]]
    c = int(const) % int(p)
    append_gidney_add_const_mod2n(qc, target, c, dirty[: max(0, n - 1)], clean3, cb, ctrl=ctrl)
    append_gidney_compare_lt_const(qc, target, c, flag, dirty, clean3, cb, ctrl=ctrl)
    # For ctrl=0 and canonical target<p, this compare is false; no explicit ctrl needed.
    append_gidney_compare_ge_const(qc, target, p, flag, dirty, clean3, cb)
    append_gidney_sub_const_mod2n(qc, target, p, dirty[: max(0, n - 1)], clean3, cb, ctrl=flag)
    append_gidney_compare_lt_const(qc, target, c, flag, dirty, clean3, cb, ctrl=ctrl)


def append_sub_const_modp_quadratic(qc: QuantumCircuit, target: Sequence[Qubit], const: int, dirty: Sequence[Qubit], aux4: Sequence[Qubit], cbits: Sequence[Clbit], *, p: int = SECP256K1_P, ctrl: Optional[Qubit] = None) -> None:
    append_add_const_modp_quadratic(qc, target, -int(const), dirty, aux4, cbits, p=p, ctrl=ctrl)


def append_neg_modp_quadratic(qc: QuantumCircuit, target: Sequence[Qubit], dirty: Sequence[Qubit], aux4: Sequence[Qubit], cbits: Sequence[Clbit], *, p: int = SECP256K1_P, ctrl: Optional[Qubit] = None) -> None:
    target = _as_list(target); dirty = _as_list(dirty); aux4 = _as_list(aux4); n = len(target)
    flag = aux4[0]; clean3 = [aux4[1], aux4[2], aux4[3]]; cb = _need_cbits(cbits, n)
    # flag ^= ctrl & (target != 0) == ctrl & (target >= 1)
    append_gidney_compare_ge_const(qc, target, 1, flag, dirty, clean3, cb, ctrl=ctrl)
    if ctrl is None:
        for q in target: qc.x(q)
        append_gidney_add_const_mod2n(qc, target, 1, dirty[: max(0, n - 1)], clean3, cb)
    else:
        for q in target: qc.cx(ctrl, q)
        append_gidney_add_const_mod2n(qc, target, 1, dirty[: max(0, n - 1)], clean3, cb, ctrl=ctrl)
    append_gidney_add_const_mod2n(qc, target, p, dirty[: max(0, n - 1)], clean3, cb, ctrl=flag)
    append_gidney_compare_ge_const(qc, target, 1, flag, dirty, clean3, cb, ctrl=ctrl)


def append_dbl_modp_quadratic(qc: QuantumCircuit, acc: Sequence[Qubit], dirty: Sequence[Qubit], aux4: Sequence[Qubit], cbits: Sequence[Clbit], *, p: int = SECP256K1_P) -> None:
    acc = _as_list(acc); dirty = _as_list(dirty); aux4 = _as_list(aux4); n = len(acc)
    if len(dirty) < n or len(aux4) < 4:
        raise ValueError("bad dbl widths")
    cb = _need_cbits(cbits, n)
    flag = aux4[0]; clean3 = [aux4[1], aux4[2], aux4[3]]
    qc.cx(acc[n - 1], flag)
    append_cyclic_left_shift(qc, acc)
    qc.cx(flag, acc[0])
    append_gidney_compare_ge_const(qc, acc, p, flag, dirty, clean3, cb)
    append_gidney_sub_const_mod2n(qc, acc, p, dirty[: max(0, n - 1)], clean3, cb, ctrl=flag)
    qc.cx(acc[0], flag)


def append_halve_modp_quadratic(qc: QuantumCircuit, acc: Sequence[Qubit], dirty: Sequence[Qubit], aux4: Sequence[Qubit], cbits: Sequence[Clbit], *, p: int = SECP256K1_P) -> None:
    acc = _as_list(acc); dirty = _as_list(dirty); aux4 = _as_list(aux4); n = len(acc)
    cb = _need_cbits(cbits, n)
    flag = aux4[0]; clean3 = [aux4[1], aux4[2], aux4[3]]
    qc.cx(acc[0], flag)
    append_gidney_add_const_mod2n(qc, acc, p, dirty[: max(0, n - 1)], clean3, cb, ctrl=flag)
    append_gidney_compare_ge_const(qc, acc, p, flag, dirty, clean3, cb)
    qc.cx(flag, acc[0])
    append_cyclic_right_shift(qc, acc)
    qc.cx(acc[n - 1], flag)


def append_mul_zero_dbladd_quadratic(qc: QuantumCircuit, x: Sequence[Qubit], y: Sequence[Qubit], out: Sequence[Qubit], aux5: Sequence[Qubit], cbits: Sequence[Clbit], *, p: int = SECP256K1_P) -> None:
    x = _as_list(x); y = _as_list(y); out = _as_list(out); aux5 = _as_list(aux5); n = len(x)
    for i in reversed(range(n)):
        if i != n - 1:
            append_dbl_modp_quadratic(qc, out, y, aux5[:4], cbits, p=p)
        append_ctrl_add_modp_quadratic(qc, x[i], y, out, aux5, cbits, p=p)


def append_mul_zero_dbladd_inverse_quadratic(qc: QuantumCircuit, x: Sequence[Qubit], y: Sequence[Qubit], out: Sequence[Qubit], aux5: Sequence[Qubit], cbits: Sequence[Clbit], *, p: int = SECP256K1_P) -> None:
    x = _as_list(x); y = _as_list(y); out = _as_list(out); aux5 = _as_list(aux5); n = len(x)
    for i in range(n):
        append_ctrl_sub_modp_quadratic(qc, x[i], y, out, aux5, cbits, p=p)
        if i != n - 1:
            append_halve_modp_quadratic(qc, out, y, aux5[:4], cbits, p=p)


def append_square_zero_dbladd_quadratic(qc: QuantumCircuit, x: Sequence[Qubit], out: Sequence[Qubit], aux6: Sequence[Qubit], cbits: Sequence[Clbit], *, p: int = SECP256K1_P) -> None:
    x = _as_list(x); out = _as_list(out); aux6 = _as_list(aux6); n = len(x)
    copied = aux6[0]; aux5 = aux6[1:6]
    for i in reversed(range(n)):
        if i != n - 1:
            append_dbl_modp_quadratic(qc, out, x, aux5[:4], cbits, p=p)
        qc.cx(x[i], copied)
        append_ctrl_add_modp_quadratic(qc, copied, x, out, aux5, cbits, p=p)
        qc.cx(x[i], copied)


def append_square_zero_dbladd_inverse_quadratic(qc: QuantumCircuit, x: Sequence[Qubit], out: Sequence[Qubit], aux6: Sequence[Qubit], cbits: Sequence[Clbit], *, p: int = SECP256K1_P) -> None:
    x = _as_list(x); out = _as_list(out); aux6 = _as_list(aux6); n = len(x)
    copied = aux6[0]; aux5 = aux6[1:6]
    for i in range(n):
        qc.cx(x[i], copied)
        append_ctrl_sub_modp_quadratic(qc, copied, x, out, aux5, cbits, p=p)
        qc.cx(x[i], copied)
        if i != n - 1:
            append_halve_modp_quadratic(qc, out, x, aux5[:4], cbits, p=p)


def _with_creg(qregs, n: int, name: str) -> QuantumCircuit:
    return QuantumCircuit(*qregs, ClassicalRegister(max(1, n), "m_arith"), name=name)


@lru_cache(maxsize=None)
def ctrl_add_modp_instruction(n: int, p: int = SECP256K1_P) -> Instruction:
    def build() -> QuantumCircuit:
        ctrl = QuantumRegister(1, "ctrl"); addend = QuantumRegister(n, "addend"); acc = QuantumRegister(n, "acc"); aux = QuantumRegister(5, "aux5")
        qc = _with_creg([ctrl, addend, acc, aux], n, f"CTRL_ADD_MODP_QUAD_DEF_{n}")
        append_ctrl_add_modp_quadratic(qc, ctrl[0], addend, acc, aux, qc.clbits, p=p)
        return qc
    return LazyDefinedInstruction(f"CTRL_ADD_MODP_QUAD_{n}", 1 + 2*n + 5, max(1, n), build)


@lru_cache(maxsize=None)
def ctrl_sub_modp_instruction(n: int, p: int = SECP256K1_P) -> Instruction:
    def build() -> QuantumCircuit:
        ctrl = QuantumRegister(1, "ctrl"); sub = QuantumRegister(n, "subtrahend"); acc = QuantumRegister(n, "acc"); aux = QuantumRegister(5, "aux5")
        qc = _with_creg([ctrl, sub, acc, aux], n, f"CTRL_SUB_MODP_QUAD_DEF_{n}")
        append_ctrl_sub_modp_quadratic(qc, ctrl[0], sub, acc, aux, qc.clbits, p=p)
        return qc
    return LazyDefinedInstruction(f"CTRL_SUB_MODP_QUAD_{n}", 1 + 2*n + 5, max(1, n), build)


@lru_cache(maxsize=None)
def add_const_modp_instruction(n: int, p: int, const: int, *, controlled: bool = False, name: Optional[str] = None) -> Instruction:
    nm = name or (f"CADD_CONST_MODP_QUAD_{n}_{const % p}" if controlled else f"ADD_CONST_MODP_QUAD_{n}_{const % p}")
    def build() -> QuantumCircuit:
        regs = []
        ctrl = None
        if controlled:
            ctrl = QuantumRegister(1, "ctrl"); regs.append(ctrl)
        target = QuantumRegister(n, "target"); dirty = QuantumRegister(n, "dirty"); aux = QuantumRegister(4, "aux4")
        qc = _with_creg(regs + [target, dirty, aux], n, nm + "_DEF")
        append_add_const_modp_quadratic(qc, target, const, dirty, aux, qc.clbits, p=p, ctrl=(ctrl[0] if ctrl else None))
        return qc
    return LazyDefinedInstruction(nm, (1 if controlled else 0) + 2*n + 4, max(1, n), build)


@lru_cache(maxsize=None)
def neg_modp_instruction(n: int, p: int, *, controlled: bool = True, name: Optional[str] = None) -> Instruction:
    nm = name or (f"CNEG_MODP_QUAD_{n}" if controlled else f"NEG_MODP_QUAD_{n}")
    def build() -> QuantumCircuit:
        regs = []
        ctrl = None
        if controlled:
            ctrl = QuantumRegister(1, "ctrl"); regs.append(ctrl)
        target = QuantumRegister(n, "target"); dirty = QuantumRegister(n, "dirty"); aux = QuantumRegister(4, "aux4")
        qc = _with_creg(regs + [target, dirty, aux], n, nm + "_DEF")
        append_neg_modp_quadratic(qc, target, dirty, aux, qc.clbits, p=p, ctrl=(ctrl[0] if ctrl else None))
        return qc
    return LazyDefinedInstruction(nm, (1 if controlled else 0) + 2*n + 4, max(1, n), build)


@lru_cache(maxsize=None)
def mul_zero_dbladd_instruction(n: int, p: int = SECP256K1_P) -> Instruction:
    def build() -> QuantumCircuit:
        x = QuantumRegister(n, "x"); y = QuantumRegister(n, "y"); out = QuantumRegister(n, "out"); aux = QuantumRegister(5, "aux5")
        qc = _with_creg([x, y, out, aux], n, f"MUL_ZERO_DBLADD_QUAD_DEF_{n}")
        append_mul_zero_dbladd_quadratic(qc, x, y, out, aux, qc.clbits, p=p)
        return qc
    return LazyDefinedInstruction(f"MUL_ZERO_DBLADD_QUAD_{n}", 3*n + 5, max(1, n), build)


@lru_cache(maxsize=None)
def mul_zero_dbladd_inverse_instruction(n: int, p: int = SECP256K1_P) -> Instruction:
    def build() -> QuantumCircuit:
        x = QuantumRegister(n, "x"); y = QuantumRegister(n, "y"); out = QuantumRegister(n, "out"); aux = QuantumRegister(5, "aux5")
        qc = _with_creg([x, y, out, aux], n, f"MUL_ZERO_DBLADD_QUAD_INV_DEF_{n}")
        append_mul_zero_dbladd_inverse_quadratic(qc, x, y, out, aux, qc.clbits, p=p)
        return qc
    return LazyDefinedInstruction(f"MUL_ZERO_DBLADD_QUAD_{n}_dg", 3*n + 5, max(1, n), build)


@lru_cache(maxsize=None)
def square_zero_dbladd_instruction(n: int, p: int = SECP256K1_P) -> Instruction:
    def build() -> QuantumCircuit:
        x = QuantumRegister(n, "x"); out = QuantumRegister(n, "out"); aux = QuantumRegister(6, "aux6")
        qc = _with_creg([x, out, aux], n, f"SQUARE_ZERO_DBLADD_QUAD_DEF_{n}")
        append_square_zero_dbladd_quadratic(qc, x, out, aux, qc.clbits, p=p)
        return qc
    return LazyDefinedInstruction(f"SQUARE_ZERO_DBLADD_QUAD_{n}", 2*n + 6, max(1, n), build)


@lru_cache(maxsize=None)
def square_zero_dbladd_inverse_instruction(n: int, p: int = SECP256K1_P) -> Instruction:
    def build() -> QuantumCircuit:
        x = QuantumRegister(n, "x"); out = QuantumRegister(n, "out"); aux = QuantumRegister(6, "aux6")
        qc = _with_creg([x, out, aux], n, f"SQUARE_ZERO_DBLADD_QUAD_INV_DEF_{n}")
        append_square_zero_dbladd_inverse_quadratic(qc, x, out, aux, qc.clbits, p=p)
        return qc
    return LazyDefinedInstruction(f"SQUARE_ZERO_DBLADD_QUAD_{n}_dg", 2*n + 6, max(1, n), build)


def build_mul_zero_dbladd_circuit(n: int = 256, p: int = SECP256K1_P) -> QuantumCircuit:
    x = QuantumRegister(n, "x"); y = QuantumRegister(n, "y"); out = QuantumRegister(n, "out"); aux = QuantumRegister(5, "aux5"); m = ClassicalRegister(max(1,n), "m_arith")
    qc = QuantumCircuit(x, y, out, aux, m, name=f"MUL_ZERO_DBLADD_QUAD_{n}")
    append_mul_zero_dbladd_quadratic(qc, x, y, out, aux, m, p=p)
    return qc


def build_square_zero_dbladd_circuit(n: int = 256, p: int = SECP256K1_P) -> QuantumCircuit:
    x = QuantumRegister(n, "x"); out = QuantumRegister(n, "out"); aux = QuantumRegister(6, "aux6"); m = ClassicalRegister(max(1,n), "m_arith")
    qc = QuantumCircuit(x, out, aux, m, name=f"SQUARE_ZERO_DBLADD_QUAD_{n}")
    append_square_zero_dbladd_quadratic(qc, x, out, aux, m, p=p)
    return qc


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--p", type=lambda s: int(s, 0), default=None)
    args = ap.parse_args()
    p = args.p if args.p is not None else ((1 << args.n) - 5 if args.n < 256 else SECP256K1_P)
    mul = build_mul_zero_dbladd_circuit(args.n, p)
    sq = build_square_zero_dbladd_circuit(args.n, p)
    print(json.dumps({
        "n": args.n,
        "p": p,
        "mul_qubits": mul.num_qubits,
        "mul_clbits": mul.num_clbits,
        "square_qubits": sq.num_qubits,
        "square_clbits": sq.num_clbits,
        "mul_top_ops": {str(k): int(v) for k, v in mul.count_ops().items()},
        "square_top_ops": {str(k): int(v) for k, v in sq.count_ops().items()},
        "backend": "gidney-constant + cuccaro-linear-compare",
        "asymptotic_toffoli": "O(n^2) for mul/square",
    }, indent=2, sort_keys=True))
