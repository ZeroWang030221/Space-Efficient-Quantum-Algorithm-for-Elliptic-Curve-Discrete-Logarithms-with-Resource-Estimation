"""Fig.15 in-place division/multiplication using the wrapped S835 FASTDUAL EEA."""
from typing import Optional, Sequence

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Clbit, Instruction, Qubit
from qiskit.circuit.library import ZGate

from under1000_eea_shared_s835_fastdual_wrapped import SECP256K1_P, eea_forward_shared_instruction, eea_inverse_shared_instruction, shared_eea_s_qubits
from quadratic_modular_arithmetic import mul_zero_dbladd_instruction, mul_zero_dbladd_inverse_instruction


def _check_widths(X: Sequence[Qubit], Y: Sequence[Qubit], A: Sequence[Qubit], S: Sequence[Qubit], b: Sequence[Clbit], m_arith: Sequence[Clbit]) -> int:
    n = len(X)
    if len(Y) != n or len(A) != n or len(b) != n:
        raise ValueError("X, Y, A and measurement b must have width n")
    if len(S) < max(shared_eea_s_qubits(n), 6):
        raise ValueError(f"S too small: need at least {max(shared_eea_s_qubits(n),6)}, got {len(S)}")
    if len(m_arith) < n:
        raise ValueError("quadratic arithmetic and wrapped EEA need n reusable classical bits")
    return n


def append_measure_reset_h_basis(qc: QuantumCircuit, reg: Sequence[Qubit], creg: Sequence[Clbit]) -> None:
    for q, c in zip(reg, creg):
        qc.h(q); qc.measure(q, c); qc.reset(q)


def append_classically_controlled_z(qc: QuantumCircuit, q: Qubit, c: Clbit) -> None:
    try:
        with qc.if_test((c, 1)):
            qc.z(q)
    except Exception:
        op = ZGate().to_mutable(); op.condition = (c, 1); qc.append(op, [q], [])


def append_z_correction_vector(qc: QuantumCircuit, reg: Sequence[Qubit], creg: Sequence[Clbit]) -> None:
    for q, c in zip(reg, creg):
        append_classically_controlled_z(qc, q, c)


def append_swap_registers(qc: QuantumCircuit, left: Sequence[Qubit], right: Sequence[Qubit]) -> None:
    for a, b in zip(left, right):
        qc.swap(a, b)


def append_inplace_division_fig15_quadratic(qc: QuantumCircuit, X: Sequence[Qubit], Y: Sequence[Qubit], A: Sequence[Qubit], S: Sequence[Qubit], b: Sequence[Clbit], m_arith: Sequence[Clbit], *, p: int = SECP256K1_P, eea_inst: Optional[Instruction] = None, eea_inv_inst: Optional[Instruction] = None) -> None:
    X = list(X); Y = list(Y); A = list(A); S = list(S); b = list(b); m_arith = list(m_arith)
    n = _check_widths(X, Y, A, S, b, m_arith)
    s_used = shared_eea_s_qubits(n)
    eea = eea_inst or eea_forward_shared_instruction(n, p)
    eea_inv = eea_inv_inst or eea_inverse_shared_instruction(n, p)
    mul = mul_zero_dbladd_instruction(n, p)
    mul_inv = mul_zero_dbladd_inverse_instruction(n, p)
    aux5 = list(S[:5])
    qc.append(eea, X + A + S[:s_used], m_arith[:n])
    qc.append(mul, X + Y + A + aux5, m_arith[:n])
    append_measure_reset_h_basis(qc, Y, b)
    qc.append(eea_inv, X + Y + S[:s_used], m_arith[:n])
    qc.append(mul, X + A + Y + aux5, m_arith[:n])
    append_z_correction_vector(qc, Y, b)
    qc.append(mul_inv, X + A + Y + aux5, m_arith[:n])
    append_swap_registers(qc, Y, A)


def append_inplace_multiplication_fig15_quadratic(qc: QuantumCircuit, X: Sequence[Qubit], Y: Sequence[Qubit], A: Sequence[Qubit], S: Sequence[Qubit], b: Sequence[Clbit], m_arith: Sequence[Clbit], *, p: int = SECP256K1_P, eea_inst: Optional[Instruction] = None, eea_inv_inst: Optional[Instruction] = None) -> None:
    X = list(X); Y = list(Y); A = list(A); S = list(S); b = list(b); m_arith = list(m_arith)
    n = _check_widths(X, Y, A, S, b, m_arith)
    s_used = shared_eea_s_qubits(n)
    eea = eea_inst or eea_forward_shared_instruction(n, p)
    eea_inv = eea_inv_inst or eea_inverse_shared_instruction(n, p)
    mul = mul_zero_dbladd_instruction(n, p)
    mul_inv = mul_zero_dbladd_inverse_instruction(n, p)
    aux5 = list(S[:5])
    qc.append(mul, X + Y + A + aux5, m_arith[:n])
    append_measure_reset_h_basis(qc, Y, b)
    qc.append(eea, X + Y + S[:s_used], m_arith[:n])
    qc.append(mul, X + A + Y + aux5, m_arith[:n])
    append_z_correction_vector(qc, Y, b)
    qc.append(mul_inv, X + A + Y + aux5, m_arith[:n])
    qc.append(eea_inv, X + Y + S[:s_used], m_arith[:n])
    append_swap_registers(qc, Y, A)


def build_inplace_division_fig15_quadratic(n: int = 256, p: int = SECP256K1_P, *, s_qubits: Optional[int] = None) -> QuantumCircuit:
    if s_qubits is None: s_qubits = shared_eea_s_qubits(n)
    X = QuantumRegister(n, "X"); Y = QuantumRegister(n, "Y"); A = QuantumRegister(n, "A"); S = QuantumRegister(s_qubits, "S_shared")
    b = ClassicalRegister(n, "b_div"); m = ClassicalRegister(max(1,n), "m_arith")
    qc = QuantumCircuit(X, Y, A, S, b, m, name=f"IDIV_FIG15_FASTDUAL_WRAPPED_QUAD_{n}")
    append_inplace_division_fig15_quadratic(qc, X, Y, A, S, b, m, p=p)
    return qc


def build_inplace_multiplication_fig15_quadratic(n: int = 256, p: int = SECP256K1_P, *, s_qubits: Optional[int] = None) -> QuantumCircuit:
    if s_qubits is None: s_qubits = shared_eea_s_qubits(n)
    X = QuantumRegister(n, "X"); Y = QuantumRegister(n, "Y"); A = QuantumRegister(n, "A"); S = QuantumRegister(s_qubits, "S_shared")
    b = ClassicalRegister(n, "b_mul"); m = ClassicalRegister(max(1,n), "m_arith")
    qc = QuantumCircuit(X, Y, A, S, b, m, name=f"IMUL_FIG15_FASTDUAL_WRAPPED_QUAD_{n}")
    append_inplace_multiplication_fig15_quadratic(qc, X, Y, A, S, b, m, p=p)
    return qc
