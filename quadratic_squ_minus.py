"""Fig.14 squ; -; squ† block using quadratic arithmetic backend."""
from typing import Sequence
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Clbit, Qubit
SECP256K1_P = (1 << 256) - (1 << 32) - 977
from quadratic_modular_arithmetic import square_zero_dbladd_instruction, square_zero_dbladd_inverse_instruction, ctrl_sub_modp_instruction


def append_squ_minus_block_quadratic(qc: QuantumCircuit, ctrl: Qubit, X: Sequence[Qubit], Y: Sequence[Qubit], A: Sequence[Qubit], S: Sequence[Qubit], m_arith: Sequence[Clbit], *, p: int = SECP256K1_P) -> None:
    X = list(X); Y = list(Y); A = list(A); S = list(S); m_arith = list(m_arith)
    n = len(X)
    if len(Y) != n or len(A) != n or len(S) < 6 or len(m_arith) < n:
        raise ValueError("bad squ-minus widths")
    sq = square_zero_dbladd_instruction(n, p)
    sq_inv = square_zero_dbladd_inverse_instruction(n, p)
    sub = ctrl_sub_modp_instruction(n, p)
    qc.append(sq, Y + A + list(S[:6]), m_arith[:n])
    qc.append(sub, [ctrl] + A + X + list(S[:5]), m_arith[:n])
    qc.append(sq_inv, Y + A + list(S[:6]), m_arith[:n])


def build_squ_minus_block_quadratic(n: int = 256, p: int = SECP256K1_P, *, s_qubits: int = 83) -> QuantumCircuit:
    ctrl = QuantumRegister(1, "ctrl"); X = QuantumRegister(n, "X"); Y = QuantumRegister(n, "Y"); A = QuantumRegister(n, "A"); S = QuantumRegister(s_qubits, "S"); m = ClassicalRegister(max(1,n), "m_arith")
    qc = QuantumCircuit(ctrl, X, Y, A, S, m, name=f"SQU_MINUS_QUAD_{n}")
    append_squ_minus_block_quadratic(qc, ctrl[0], X, Y, A, S, m, p=p)
    return qc
