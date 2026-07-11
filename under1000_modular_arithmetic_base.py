from functools import lru_cache
from typing import Iterable, Optional, Sequence

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate, Instruction, Qubit

SECP256K1_P = (1 << 256) - (1 << 32) - 977


def _mask(n: int) -> int:
    return (1 << n) - 1


def _as_list(xs: Sequence[Qubit]) -> list[Qubit]:
    return list(xs)


def _append_cx_multi(qc: QuantumCircuit, controls: Sequence[Qubit], target: Qubit) -> None:
    """Toggle target under all controls, using no explicit ancilla qubits."""
    controls = list(controls)
    if len(controls) == 0:
        qc.x(target)
    elif len(controls) == 1:
        qc.cx(controls[0], target)
    elif len(controls) == 2:
        qc.ccx(controls[0], controls[1], target)
    else:
        qc.mcx(controls, target)


def _with_literal_controls(
    qc: QuantumCircuit,
    literals: Sequence[tuple[Qubit, int]],
    target: Qubit,
) -> None:
    """Toggle target if all literal controls match.  Qubits are restored."""
    controls: list[Qubit] = []
    for q, bit in literals:
        if bit not in (0, 1):
            raise ValueError("literal bit must be 0 or 1")
        if bit == 0:
            qc.x(q)
        controls.append(q)
    _append_cx_multi(qc, controls, target)
    for q, bit in reversed(list(literals)):
        if bit == 0:
            qc.x(q)


def append_const_add_mod2n(
    qc: QuantumCircuit,
    reg: Sequence[Qubit],
    const: int,
    *,
    ctrl: Optional[Qubit] = None,
) -> None:
    """Exact in-place ``reg += ctrl*const (mod 2^n)`` using no ancillas.

    The construction adds each set bit 2^i by incrementing the suffix reg[i:].
    A suffix increment is a descending chain of multi-controlled X gates.
    """
    reg = _as_list(reg)
    n = len(reg)
    c = int(const) & _mask(n)
    if n == 0 or c == 0:
        return
    prefix_ctrl = [ctrl] if ctrl is not None else []
    for i in range(n):
        if ((c >> i) & 1) == 0:
            continue
        # Increment the little-endian suffix reg[i:n].
        for j in range(n - 1, i, -1):
            _append_cx_multi(qc, prefix_ctrl + reg[i:j], reg[j])
        if ctrl is None:
            qc.x(reg[i])
        else:
            qc.cx(ctrl, reg[i])


def append_const_sub_mod2n(qc: QuantumCircuit, reg: Sequence[Qubit], const: int, *, ctrl: Optional[Qubit] = None) -> None:
    append_const_add_mod2n(qc, reg, -int(const), ctrl=ctrl)


def append_xor_ge_const(
    qc: QuantumCircuit,
    reg: Sequence[Qubit],
    const: int,
    flag: Qubit,
    *,
    ctrl: Optional[Qubit] = None,
) -> None:
    """``flag ^= ctrl & (reg >= const)`` for little-endian ``reg``.

    Implemented as a disjoint lexicographic comparison against const-1.
    """
    reg = _as_list(reg)
    n = len(reg)
    c = int(const)
    if c <= 0:
        if ctrl is None:
            qc.x(flag)
        else:
            qc.cx(ctrl, flag)
        return
    if c >= (1 << n):
        return
    threshold = c - 1
    # reg > threshold.  For the highest differing bit i, reg_i=1 and threshold_i=0.
    for i in reversed(range(n)):
        if ((threshold >> i) & 1) != 0:
            continue
        literals: list[tuple[Qubit, int]] = []
        if ctrl is not None:
            literals.append((ctrl, 1))
        for h in range(n - 1, i, -1):
            literals.append((reg[h], (threshold >> h) & 1))
        literals.append((reg[i], 1))
        _with_literal_controls(qc, literals, flag)


def append_xor_lt_const(
    qc: QuantumCircuit,
    reg: Sequence[Qubit],
    const: int,
    flag: Qubit,
    *,
    ctrl: Optional[Qubit] = None,
) -> None:
    """``flag ^= ctrl & (reg < const)`` for little-endian ``reg``."""
    reg = _as_list(reg)
    n = len(reg)
    c = int(const)
    if c <= 0:
        return
    if c >= (1 << n):
        if ctrl is None:
            qc.x(flag)
        else:
            qc.cx(ctrl, flag)
        return
    # First differing bit has reg_i=0 and const_i=1.
    for i in reversed(range(n)):
        if ((c >> i) & 1) == 0:
            continue
        literals: list[tuple[Qubit, int]] = []
        if ctrl is not None:
            literals.append((ctrl, 1))
        for h in range(n - 1, i, -1):
            literals.append((reg[h], (c >> h) & 1))
        literals.append((reg[i], 0))
        _with_literal_controls(qc, literals, flag)


def append_xor_nonzero(qc: QuantumCircuit, reg: Sequence[Qubit], flag: Qubit, *, ctrl: Optional[Qubit] = None) -> None:
    append_xor_ge_const(qc, reg, 1, flag, ctrl=ctrl)


def append_xor_lt_quantum(
    qc: QuantumCircuit,
    left: Sequence[Qubit],
    right: Sequence[Qubit],
    flag: Qubit,
    *,
    ctrl: Optional[Qubit] = None,
) -> None:
    """``flag ^= ctrl & (left < right)`` preserving both registers.

    For each candidate highest differing bit i, temporarily maps right[h] to
    right[h] xor left[h] for h>i, uses negative controls on these equality bits,
    and then restores them.
    """
    left = _as_list(left)
    right = _as_list(right)
    if len(left) != len(right):
        raise ValueError("comparison registers must have equal width")
    n = len(left)
    for i in reversed(range(n)):
        for h in range(i + 1, n):
            qc.cx(left[h], right[h])
        literals: list[tuple[Qubit, int]] = []
        if ctrl is not None:
            literals.append((ctrl, 1))
        for h in range(n - 1, i, -1):
            literals.append((right[h], 0))  # equality marker
        literals.append((left[i], 0))
        literals.append((right[i], 1))
        _with_literal_controls(qc, literals, flag)
        for h in reversed(range(i + 1, n)):
            qc.cx(left[h], right[h])


def append_controlled_add_mod2_with_carry_flag(
    qc: QuantumCircuit,
    ctrl: Qubit,
    addend: Sequence[Qubit],
    acc: Sequence[Qubit],
    carry: Qubit,
    carry_flag: Qubit,
) -> None:
    """Controlled add mod 2^n and xor the carry-out into ``carry_flag``.

    This is the same Cuccaro-style controlled adder as
    ``append_controlled_add_mod2_3n``, with one extra readout of the carry between
    the forward and reverse passes.  ``carry`` is restored to |0>.
    """
    b = _as_list(addend)
    a = _as_list(acc)
    n = len(a)
    if len(b) != n:
        raise ValueError("addend and acc widths differ")
    if n == 0:
        return
    if n == 1:
        qc.ccx(ctrl, b[0], a[0])
        qc.ccx(ctrl, b[0], carry_flag)
        return
    for i in range(n):
        qc.cx(carry, b[i])
        qc.cx(carry, a[i])
        qc.ccx(a[i], b[i], carry)
    qc.ccx(ctrl, carry, carry_flag)
    for i in reversed(range(n)):
        qc.ccx(a[i], b[i], carry)
        qc.cx(carry, a[i])
        qc.ccx(ctrl, b[i], a[i])
        qc.cx(carry, b[i])


@lru_cache(maxsize=None)
def ctrl_add_modp_gate(n: int, p: int = SECP256K1_P) -> Gate:
    ctrl = QuantumRegister(1, "ctrl")
    addend = QuantumRegister(n, "addend")
    acc = QuantumRegister(n, "acc")
    aux = QuantumRegister(2, "aux")  # carry, reduction flag
    qc = QuantumCircuit(ctrl, addend, acc, aux, name=f"CTRL_ADD_MODP_REAL_{n}")
    carry, flag = aux[0], aux[1]
    append_controlled_add_mod2_with_carry_flag(qc, ctrl[0], addend, acc, carry, flag)
    append_xor_ge_const(qc, acc, p, flag, ctrl=ctrl[0])
    append_const_add_mod2n(qc, acc, -p, ctrl=flag)
    append_xor_lt_quantum(qc, acc, addend, flag, ctrl=ctrl[0])
    return qc.to_gate()


@lru_cache(maxsize=None)
def ctrl_sub_modp_gate(n: int, p: int = SECP256K1_P) -> Gate:
    return ctrl_add_modp_gate(n, p).inverse()


@lru_cache(maxsize=None)
def add_const_modp_gate(n: int, p: int, const: int, *, controlled: bool = False, name: Optional[str] = None) -> Gate:
    regs = []
    ctrl = None
    if controlled:
        ctrl = QuantumRegister(1, "ctrl")
        regs.append(ctrl)
    target = QuantumRegister(n, "target")
    aux = QuantumRegister(1, "flag")
    regs += [target, aux]
    qc = QuantumCircuit(*regs, name=name or (f"CADD_CONST_MODP_REAL_{n}" if controlled else f"ADD_CONST_MODP_REAL_{n}"))
    c = int(const) % int(p)
    flag = aux[0]
    cctrl = ctrl[0] if controlled else None
    append_const_add_mod2n(qc, target, c, ctrl=cctrl)
    append_xor_lt_const(qc, target, c, flag, ctrl=cctrl)
    append_xor_ge_const(qc, target, p, flag, ctrl=cctrl)
    append_const_add_mod2n(qc, target, -p, ctrl=flag)
    append_xor_lt_const(qc, target, c, flag, ctrl=cctrl)
    return qc.to_gate()


@lru_cache(maxsize=None)
def neg_modp_gate(n: int, p: int, *, controlled: bool = True, name: Optional[str] = None) -> Gate:
    regs = []
    ctrl = None
    if controlled:
        ctrl = QuantumRegister(1, "ctrl")
        regs.append(ctrl)
    target = QuantumRegister(n, "target")
    flag = QuantumRegister(1, "flag")
    regs += [target, flag]
    qc = QuantumCircuit(*regs, name=name or (f"CNEG_MODP_REAL_{n}" if controlled else f"NEG_MODP_REAL_{n}"))
    cctrl = ctrl[0] if controlled else None
    append_xor_nonzero(qc, target, flag[0], ctrl=cctrl)
    if controlled:
        for q in target:
            qc.cx(ctrl[0], q)
        append_const_add_mod2n(qc, target, 1, ctrl=ctrl[0])
    else:
        for q in target:
            qc.x(q)
        append_const_add_mod2n(qc, target, 1)
    append_const_add_mod2n(qc, target, p, ctrl=flag[0])
    append_xor_nonzero(qc, target, flag[0], ctrl=cctrl)
    return qc.to_gate()


@lru_cache(maxsize=None)
def dbl_modp_gate(n: int, p: int = SECP256K1_P) -> Gate:
    acc = QuantumRegister(n, "acc")
    flag = QuantumRegister(1, "flag")
    qc = QuantumCircuit(acc, flag, name=f"DBL_MODP_REAL_{n}")
    if n == 0:
        return qc.to_gate()
    qc.cx(acc[n - 1], flag[0])
    for i in reversed(range(n - 1)):
        qc.swap(acc[i], acc[i + 1])
    qc.cx(flag[0], acc[0])
    append_xor_ge_const(qc, acc, p, flag[0])
    append_const_add_mod2n(qc, acc, -p, ctrl=flag[0])
    qc.cx(acc[0], flag[0])
    return qc.to_gate()


@lru_cache(maxsize=None)
def halve_modp_gate(n: int, p: int = SECP256K1_P) -> Gate:
    return dbl_modp_gate(n, p).inverse()


def append_mul_zero_dbladd_real(qc: QuantumCircuit, x: Sequence[Qubit], y: Sequence[Qubit], out: Sequence[Qubit], aux2: Sequence[Qubit], *, p: int) -> None:
    x = _as_list(x); y = _as_list(y); out = _as_list(out); aux2 = _as_list(aux2)
    n = len(x)
    if len(y) != n or len(out) != n or len(aux2) < 2:
        raise ValueError("bad mul widths")
    add = ctrl_add_modp_gate(n, p)
    dbl = dbl_modp_gate(n, p)
    for i in reversed(range(n)):
        if i != n - 1:
            qc.append(dbl, out + [aux2[0]])
        qc.append(add, [x[i]] + y + out + aux2[:2])


def append_mul_zero_dbladd_inverse_real(qc: QuantumCircuit, x: Sequence[Qubit], y: Sequence[Qubit], out: Sequence[Qubit], aux2: Sequence[Qubit], *, p: int) -> None:
    x = _as_list(x); y = _as_list(y); out = _as_list(out); aux2 = _as_list(aux2)
    n = len(x)
    add_inv = ctrl_sub_modp_gate(n, p)
    half = halve_modp_gate(n, p)
    for i in range(n):
        qc.append(add_inv, [x[i]] + y + out + aux2[:2])
        if i != n - 1:
            qc.append(half, out + [aux2[0]])


@lru_cache(maxsize=None)
def mul_zero_dbladd_gate(n: int, p: int = SECP256K1_P) -> Gate:
    x = QuantumRegister(n, "x")
    y = QuantumRegister(n, "y")
    out = QuantumRegister(n, "out")
    aux = QuantumRegister(2, "aux")
    qc = QuantumCircuit(x, y, out, aux, name=f"MUL_ZERO_DBLADD_REAL_{n}")
    append_mul_zero_dbladd_real(qc, x, y, out, aux, p=p)
    return qc.to_gate()


@lru_cache(maxsize=None)
def mul_zero_dbladd_inverse_gate(n: int, p: int = SECP256K1_P) -> Gate:
    # Use the exact inverse of the full unitary definition.
    return mul_zero_dbladd_gate(n, p).inverse()


def append_square_zero_dbladd_real(qc: QuantumCircuit, x: Sequence[Qubit], out: Sequence[Qubit], aux3: Sequence[Qubit], *, p: int) -> None:
    x = _as_list(x); out = _as_list(out); aux3 = _as_list(aux3)
    n = len(x)
    if len(out) != n or len(aux3) < 3:
        raise ValueError("bad square widths")
    copied = aux3[0]
    aux2 = aux3[1:3]
    add = ctrl_add_modp_gate(n, p)
    dbl = dbl_modp_gate(n, p)
    for i in reversed(range(n)):
        if i != n - 1:
            qc.append(dbl, out + [aux2[0]])
        qc.cx(x[i], copied)
        qc.append(add, [copied] + x + out + aux2)
        qc.cx(x[i], copied)


@lru_cache(maxsize=None)
def square_zero_dbladd_gate(n: int, p: int = SECP256K1_P) -> Gate:
    x = QuantumRegister(n, "x")
    out = QuantumRegister(n, "out")
    aux = QuantumRegister(3, "aux")
    qc = QuantumCircuit(x, out, aux, name=f"SQUARE_ZERO_DBLADD_REAL_{n}")
    append_square_zero_dbladd_real(qc, x, out, aux, p=p)
    return qc.to_gate()


@lru_cache(maxsize=None)
def square_zero_dbladd_inverse_gate(n: int, p: int = SECP256K1_P) -> Gate:
    return square_zero_dbladd_gate(n, p).inverse()


def build_mul_zero_dbladd_circuit(n: int = 256, p: int = SECP256K1_P) -> QuantumCircuit:
    x = QuantumRegister(n, "x")
    y = QuantumRegister(n, "y")
    out = QuantumRegister(n, "out")
    aux = QuantumRegister(2, "aux")
    qc = QuantumCircuit(x, y, out, aux, name=f"MUL_ZERO_DBLADD_REAL_{n}")
    append_mul_zero_dbladd_real(qc, x, y, out, aux, p=p)
    return qc


def build_square_zero_dbladd_circuit(n: int = 256, p: int = SECP256K1_P) -> QuantumCircuit:
    x = QuantumRegister(n, "x")
    out = QuantumRegister(n, "out")
    aux = QuantumRegister(3, "aux")
    qc = QuantumCircuit(x, out, aux, name=f"SQUARE_ZERO_DBLADD_REAL_{n}")
    append_square_zero_dbladd_real(qc, x, out, aux, p=p)
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
        "square_qubits": sq.num_qubits,
        "mul_top_ops": {str(k): int(v) for k, v in mul.count_ops().items()},
        "square_top_ops": {str(k): int(v) for k, v in sq.count_ops().items()},
        "uses_definitionless_placeholders": False,
    }, indent=2, sort_keys=True))
