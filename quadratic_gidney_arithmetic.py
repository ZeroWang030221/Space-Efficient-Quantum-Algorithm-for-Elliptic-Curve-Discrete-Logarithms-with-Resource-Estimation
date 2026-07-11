"""Linear-Toffoli arithmetic primitives for the under-1000 point-addition build.

This module ports the hot-path ideas from Gidney's 2025 constant-workspace
classical-quantum adder and the ecdsafail challenge implementation to Qiskit:

* constant add/sub mod 2^n with 3 clean qubits and n-1 borrowed dirty qubits;
* constant comparison a >= k with 3 clean qubits and n borrowed dirty qubits;
* controlled variants whose constant bits are gated by a control qubit;
* Cuccaro high-bit quantum comparator for a < b using one clean carry qubit.

The Gidney primitives are dynamic circuits: carry qubits are H-basis measured,
reset, and corrected later by classically controlled Z gates on borrowed dirty
qubits.  The caller supplies a classical register of length at least n.  These
classical bits are not quantum width.
"""
from typing import Optional, Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import Clbit, Qubit
from qiskit.circuit.library import ZGate, CZGate


def _bit(c: int, i: int) -> int:
    return (int(c) >> int(i)) & 1


def _as_list(xs: Sequence[Qubit]) -> list[Qubit]:
    return list(xs)


def _cond_z(qc: QuantumCircuit, c: Clbit, q: Qubit) -> None:
    op = ZGate().to_mutable()
    op.condition = (c, 1)
    qc.append(op, [q], [])


def _cond_cz(qc: QuantumCircuit, c: Clbit, a: Qubit, b: Qubit) -> None:
    op = CZGate().to_mutable()
    op.condition = (c, 1)
    qc.append(op, [a, b], [])


def _hmr(qc: QuantumCircuit, q: Qubit, c: Clbit) -> None:
    qc.h(q)
    qc.measure(q, c)
    qc.reset(q)


def _ccx_cond_const(qc: QuantumCircuit, c1: Qubit, c2: Qubit, target: Qubit, b0: int, b1: int, ctrl: Optional[Qubit] = None) -> None:
    """target ^= (c1 xor b0) & (c2 xor b1), optionally only when ctrl=1.

    For ctrl=None, b0/b1 are ordinary compile-time inverted controls.
    For ctrl!=None, the inversions are themselves gated by ctrl, matching the
    controlled Gidney carry-xor trick: when ctrl=0, the constant is zero.
    """
    if ctrl is None:
        if b0: qc.x(c1)
        if b1: qc.x(c2)
        qc.ccx(c1, c2, target)
        if b1: qc.x(c2)
        if b0: qc.x(c1)
    else:
        if b0: qc.cx(ctrl, c1)
        if b1: qc.cx(ctrl, c2)
        qc.ccx(c1, c2, target)
        if b1: qc.cx(ctrl, c2)
        if b0: qc.cx(ctrl, c1)


def _xor_carries(qc: QuantumCircuit, a: Sequence[Qubit], const: int, out: Sequence[Qubit], clean: Sequence[Qubit], *, ctrl: Optional[Qubit] = None) -> None:
    """out[0:n-1] ^= carries[1:n] of a + ctrl*const.

    This is the involutory XORCarries core used to restore dirty bits in the
    Gidney constant adder.  The register a is the complemented final sum, as in
    the original construction.
    """
    a = _as_list(a); out = _as_list(out); clean = _as_list(clean)
    n = len(a)
    if n <= 1:
        return
    if len(out) < n - 1:
        raise ValueError("xor_carries needs n-1 dirty/output bits")
    if len(clean) < 1:
        raise ValueError("xor_carries needs one clean qubit")
    cin = clean[0]
    for i in range(n - 2, 0, -1):
        _ccx_cond_const(qc, a[i], out[i - 1], out[i], _bit(const, i), 0, ctrl=ctrl)
    for i in range(n - 1):
        if _bit(const, i):
            if ctrl is None:
                qc.x(out[i])
            else:
                qc.cx(ctrl, out[i])
    _ccx_cond_const(qc, cin, a[0], out[0], _bit(const, 0), _bit(const, 0), ctrl=ctrl)
    for i in range(1, n - 1):
        _ccx_cond_const(qc, a[i], out[i - 1], out[i], _bit(const, i), _bit(const, i), ctrl=ctrl)


def _xor_carries_all(qc: QuantumCircuit, a: Sequence[Qubit], const: int, out: Sequence[Qubit], clean: Sequence[Qubit], *, ctrl: Optional[Qubit] = None) -> None:
    """out[0:n] ^= carries[1:n+1] of a + ctrl*const, including overflow."""
    a = _as_list(a); out = _as_list(out); clean = _as_list(clean)
    n = len(a)
    if len(out) < n:
        raise ValueError("xor_carries_all needs n dirty/output bits")
    if len(clean) < 1:
        raise ValueError("xor_carries_all needs one clean qubit")
    cin = clean[0]
    for i in range(n - 1, 0, -1):
        _ccx_cond_const(qc, a[i], out[i - 1], out[i], _bit(const, i), 0, ctrl=ctrl)
    for i in range(n):
        if _bit(const, i):
            if ctrl is None:
                qc.x(out[i])
            else:
                qc.cx(ctrl, out[i])
    _ccx_cond_const(qc, cin, a[0], out[0], _bit(const, 0), _bit(const, 0), ctrl=ctrl)
    for i in range(1, n):
        _ccx_cond_const(qc, a[i], out[i - 1], out[i], _bit(const, i), _bit(const, i), ctrl=ctrl)


def append_gidney_add_const_mod2n(
    qc: QuantumCircuit,
    target: Sequence[Qubit],
    const: int,
    dirty: Sequence[Qubit],
    clean: Sequence[Qubit],
    cbits: Sequence[Clbit],
    *,
    ctrl: Optional[Qubit] = None,
) -> None:
    """target += ctrl*const (mod 2^n), restoring dirty/clean.

    Requires len(dirty)>=n-1, len(clean)>=3, len(cbits)>=n-1.  When ctrl is
    None, the addition is unconditional.  With ctrl, the constant bits are gated
    by ctrl; the controlled variant has the same Toffoli order.
    """
    a = _as_list(target); dirty = _as_list(dirty); clean = _as_list(clean); cbits = list(cbits)
    n = len(a)
    const &= (1 << n) - 1 if n else 0
    if n == 0 or const == 0:
        return
    if n == 1:
        if _bit(const, 0):
            qc.cx(ctrl, a[0]) if ctrl is not None else qc.x(a[0])
        return
    if len(dirty) < n - 1:
        raise ValueError(f"Gidney constant adder needs {n-1} dirty qubits")
    if len(clean) < 3:
        raise ValueError("Gidney constant adder needs 3 clean qubits")
    if len(cbits) < n - 1:
        raise ValueError(f"Gidney constant adder needs {n-1} classical bits")

    cy, nxt, anc = clean[0], clean[1], clean[2]
    cur, spare = cy, nxt
    for i in range(n - 1):
        if _bit(const, i):
            qc.cx(ctrl, anc) if ctrl is not None else qc.x(anc)
        qc.cx(cur, anc)       # anc = c_i XOR carry_i
        qc.cx(cur, a[i])      # target bit becomes target_i XOR carry_i
        qc.ccx(a[i], anc, spare)
        qc.cx(cur, spare)     # spare = carry_{i+1}
        qc.cx(spare, dirty[i])
        qc.cx(cur, anc)
        if _bit(const, i):
            if ctrl is None:
                qc.x(anc); qc.x(a[i])
            else:
                qc.cx(ctrl, anc); qc.cx(ctrl, a[i])
        if i > 0:
            _hmr(qc, cur, cbits[i - 1])
        cur, spare = spare, cur
    if _bit(const, n - 1):
        qc.cx(ctrl, a[n - 1]) if ctrl is not None else qc.x(a[n - 1])
    qc.cx(cur, a[n - 1])
    _hmr(qc, cur, cbits[n - 2])

    # First phase correction: dirty contains dirty_orig XOR carry.
    for i in range(n - 1):
        _cond_z(qc, cbits[i], dirty[i])

    # Restore dirty using the complemented output sum.
    for q in a:
        qc.x(q)
    _xor_carries(qc, a, const, dirty[: n - 1], clean, ctrl=ctrl)
    for q in a:
        qc.x(q)

    # Second phase correction: dirty is back to dirty_orig.
    for i in range(n - 1):
        _cond_z(qc, cbits[i], dirty[i])


def append_gidney_sub_const_mod2n(qc: QuantumCircuit, target: Sequence[Qubit], const: int, dirty: Sequence[Qubit], clean: Sequence[Qubit], cbits: Sequence[Clbit], *, ctrl: Optional[Qubit] = None) -> None:
    append_gidney_add_const_mod2n(qc, target, -int(const), dirty, clean, cbits, ctrl=ctrl)


def append_gidney_compare_ge_const(
    qc: QuantumCircuit,
    target: Sequence[Qubit],
    k: int,
    out: Qubit,
    dirty: Sequence[Qubit],
    clean: Sequence[Qubit],
    cbits: Sequence[Clbit],
    *,
    ctrl: Optional[Qubit] = None,
) -> None:
    """out ^= ctrl & [target >= k], restoring target/dirty/clean.

    The comparison is the overflow carry of target + (2^n-k).  It uses n
    dirty borrowed qubits to hold carry_1..carry_n temporarily.
    """
    a = _as_list(target); dirty = _as_list(dirty); clean = _as_list(clean); cbits = list(cbits)
    n = len(a)
    k = int(k)
    if n == 0:
        if k <= 0:
            qc.cx(ctrl, out) if ctrl is not None else qc.x(out)
        return
    if k <= 0:
        qc.cx(ctrl, out) if ctrl is not None else qc.x(out)
        return
    if k >= (1 << n):
        return
    if len(dirty) < n:
        raise ValueError(f"Gidney comparator needs {n} dirty qubits")
    if len(clean) < 3:
        raise ValueError("Gidney comparator needs 3 clean qubits")
    if len(cbits) < n:
        raise ValueError(f"Gidney comparator needs {n} classical bits")
    const = ((1 << n) - k) & ((1 << n) - 1)
    cy, nxt, anc = clean[0], clean[1], clean[2]
    cur, spare = cy, nxt
    for i in range(n):
        if _bit(const, i):
            qc.cx(ctrl, anc) if ctrl is not None else qc.x(anc)
        qc.cx(cur, anc)
        qc.cx(cur, a[i])
        qc.ccx(a[i], anc, spare)
        qc.cx(cur, spare)
        qc.cx(spare, dirty[i])
        if i == n - 1:
            qc.cx(spare, out)  # carry_n
        qc.cx(cur, a[i])       # restore target bit instead of forming sum
        qc.cx(cur, anc)
        if _bit(const, i):
            qc.cx(ctrl, anc) if ctrl is not None else qc.x(anc)
        if i > 0:
            _hmr(qc, cur, cbits[i - 1])
        cur, spare = spare, cur
    _hmr(qc, cur, cbits[n - 1])

    for i in range(n):
        _cond_z(qc, cbits[i], dirty[i])
    _xor_carries_all(qc, a, const, dirty[:n], clean, ctrl=ctrl)
    for i in range(n):
        _cond_z(qc, cbits[i], dirty[i])


def append_gidney_compare_lt_const(qc: QuantumCircuit, target: Sequence[Qubit], k: int, out: Qubit, dirty: Sequence[Qubit], clean: Sequence[Qubit], cbits: Sequence[Clbit], *, ctrl: Optional[Qubit] = None) -> None:
    """out ^= ctrl & [target < k], restoring all work."""
    n = len(target)
    if k <= 0:
        return
    if k >= (1 << n):
        qc.cx(ctrl, out) if ctrl is not None else qc.x(out)
        return
    # [a < k] = NOT [a >= k]
    qc.cx(ctrl, out) if ctrl is not None else qc.x(out)
    append_gidney_compare_ge_const(qc, target, k, out, dirty, clean, cbits, ctrl=ctrl)


def _maj(qc: QuantumCircuit, a: Qubit, b: Qubit, c: Qubit) -> None:
    qc.cx(a, b); qc.cx(a, c); qc.ccx(c, b, a)


def _maj_inv(qc: QuantumCircuit, a: Qubit, b: Qubit, c: Qubit) -> None:
    qc.ccx(c, b, a); qc.cx(a, c); qc.cx(a, b)


def append_xor_lt_quantum_cuccaro(qc: QuantumCircuit, left: Sequence[Qubit], right: Sequence[Qubit], flag: Qubit, carry: Qubit, *, ctrl: Optional[Qubit] = None) -> None:
    """flag ^= ctrl & [left < right], preserving left/right and resetting carry.

    Implements the high-bit-only Cuccaro comparator by computing the carry-out of
    left + (~right) + 1.  This is linear in n and uses one clean carry bit.
    """
    left = _as_list(left); right = _as_list(right)
    n = len(left)
    if len(right) != n:
        raise ValueError("comparison widths differ")
    if n == 0:
        return
    if n == 1:
        qc.x(left[0])
        if ctrl is None:
            qc.ccx(left[0], right[0], flag)
        else:
            qc.ccx(left[0], right[0], carry)
            qc.ccx(ctrl, carry, flag)
            qc.ccx(left[0], right[0], carry)
        qc.x(left[0])
        return
    for q in right:
        qc.x(q)
    qc.x(carry)  # incoming carry 1
    _maj(qc, left[0], right[0], carry)
    for i in range(1, n):
        _maj(qc, left[i], right[i], left[i - 1])
    # left[n-1] is the carry-out; left < right iff this bit is 0.
    qc.x(left[n - 1])
    if ctrl is None:
        qc.cx(left[n - 1], flag)
    else:
        qc.ccx(ctrl, left[n - 1], flag)
    qc.x(left[n - 1])
    for i in reversed(range(1, n)):
        _maj_inv(qc, left[i], right[i], left[i - 1])
    _maj_inv(qc, left[0], right[0], carry)
    qc.x(carry)
    for q in right:
        qc.x(q)


def append_controlled_add_mod2_with_carry_flag(qc: QuantumCircuit, ctrl: Qubit, addend: Sequence[Qubit], acc: Sequence[Qubit], carry: Qubit, carry_flag: Qubit) -> None:
    """acc += ctrl*addend mod 2^n and carry_flag ^= carry_out; carry reset."""
    b = _as_list(addend); a = _as_list(acc); n = len(a)
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


def append_controlled_sub_mod2_with_carry_flag(qc: QuantumCircuit, ctrl: Qubit, addend: Sequence[Qubit], acc: Sequence[Qubit], carry: Qubit, carry_flag: Qubit) -> None:
    """Inverse of append_controlled_add_mod2_with_carry_flag."""
    # Build by literal reverse of the gate sequence above.  The carry flag is
    # toggled by the same carry-out of the forward add before the add is undone.
    b = _as_list(addend); a = _as_list(acc); n = len(a)
    if n == 0:
        return
    if n == 1:
        qc.ccx(ctrl, b[0], carry_flag)
        qc.ccx(ctrl, b[0], a[0])
        return
    for i in range(n):
        qc.cx(carry, b[i])
        qc.ccx(ctrl, b[i], a[i])
        qc.cx(carry, a[i])
        qc.ccx(a[i], b[i], carry)
    qc.ccx(ctrl, carry, carry_flag)
    for i in reversed(range(n)):
        qc.ccx(a[i], b[i], carry)
        qc.cx(carry, a[i])
        qc.cx(carry, b[i])
