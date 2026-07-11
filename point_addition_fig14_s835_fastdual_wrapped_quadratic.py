"""Corrected S835 FASTDUAL Fig.14 point addition with full Algorithm-1 EEA wrapper.

The live quantum registers are exactly ctrl + X + Y + A + S.  For n=256,
S=66 and the quantum width is 835.  The EEA block includes the Algorithm-1
pre/postprocessing wrapper; fixed-constant arithmetic uses the quadratic
Gidney/Cuccaro backend.  Classical bits are used only for measurement-based
feed-forward and are not counted as logical qubits.
"""
import argparse, json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit

from under1000_eea_shared_s835_fastdual_wrapped import SECP256K1_P, shared_eea_layout, shared_eea_s_qubits
from quadratic_modular_arithmetic import add_const_modp_instruction, neg_modp_instruction
from quadratic_fig15_inplace_s835_fastdual_wrapped import append_inplace_division_fig15_quadratic, append_inplace_multiplication_fig15_quadratic
from quadratic_squ_minus import append_squ_minus_block_quadratic


@dataclass(frozen=True)
class WrappedPointAdditionBudget:
    n: int
    s_qubits: int
    include_control: bool = True
    @property
    def control_qubits(self) -> int: return 1 if self.include_control else 0
    @property
    def field_register_qubits(self) -> int: return 3 * self.n
    @property
    def total_quantum_qubits(self) -> int: return self.control_qubits + self.field_register_qubits + self.s_qubits
    def as_dict(self) -> dict:
        d = asdict(self)
        d.update({
            "control_qubits": self.control_qubits,
            "field_register_qubits": self.field_register_qubits,
            "total_quantum_qubits": self.total_quantum_qubits,
            "under_1000": self.total_quantum_qubits < 1000,
        })
        return d


def _append_add_const(qc: QuantumCircuit, *, reg: Sequence[Qubit], dirty: Sequence[Qubit], ctrl: Optional[Qubit], S: Sequence[Qubit], m_arith, n: int, p: int, const: int, name: str) -> None:
    controlled = ctrl is not None
    inst = add_const_modp_instruction(n, p, const, controlled=controlled, name=name)
    qargs = ([ctrl] if controlled else []) + list(reg) + list(dirty) + list(S[:4])
    qc.append(inst, qargs, list(m_arith)[:n])


def build_point_addition_fig14_quadratic(*, n: int = 256, p: int = SECP256K1_P, x2: int = 0, y2: int = 0, s_qubits: Optional[int] = None, name: str = "POINT_ADDITION_FIG14_FASTDUAL_WRAPPED_QUADRATIC") -> QuantumCircuit:
    if s_qubits is None:
        s_qubits = shared_eea_s_qubits(n)
    if s_qubits < max(shared_eea_s_qubits(n), 6):
        raise ValueError(f"S too small: need {max(shared_eea_s_qubits(n),6)}, got {s_qubits}")
    ctrl = QuantumRegister(1, "ctrl")
    X = QuantumRegister(n, "X_x1_to_x3")
    Y = QuantumRegister(n, "Y_y1_to_y3")
    A = QuantumRegister(n, "A_shared_work")
    S = QuantumRegister(s_qubits, "S_shared_eea_arith")
    b_div = ClassicalRegister(n, "b_div")
    b_mul = ClassicalRegister(n, "b_mul")
    m_arith = ClassicalRegister(max(1,n), "m_arith")
    qc = QuantumCircuit(ctrl, X, Y, A, S, b_div, b_mul, m_arith, name=f"{name}_{n}")

    _append_add_const(qc, reg=X, dirty=A, ctrl=None, S=S, m_arith=m_arith, n=n, p=p, const=-x2, name="QUAD_X_SUB_X2")
    _append_add_const(qc, reg=Y, dirty=A, ctrl=ctrl[0], S=S, m_arith=m_arith, n=n, p=p, const=-y2, name="QUAD_CTRL_Y_SUB_Y2")
    append_inplace_division_fig15_quadratic(qc, X, Y, A, S, b_div, m_arith, p=p)
    append_squ_minus_block_quadratic(qc, ctrl[0], X, Y, A, S, m_arith, p=p)
    _append_add_const(qc, reg=X, dirty=A, ctrl=ctrl[0], S=S, m_arith=m_arith, n=n, p=p, const=3*x2, name="QUAD_CTRL_X_ADD_3X2")
    append_inplace_multiplication_fig15_quadratic(qc, X, Y, A, S, b_mul, m_arith, p=p)
    neg = neg_modp_instruction(n, p, controlled=True, name="QUAD_CTRL_NEG_X")
    qc.append(neg, [ctrl[0]] + list(X) + list(A) + list(S[:4]), list(m_arith)[:n])
    _append_add_const(qc, reg=X, dirty=A, ctrl=None, S=S, m_arith=m_arith, n=n, p=p, const=x2, name="QUAD_X_ADD_X2")
    _append_add_const(qc, reg=Y, dirty=A, ctrl=ctrl[0], S=S, m_arith=m_arith, n=n, p=p, const=-y2, name="QUAD_CTRL_Y_SUB_Y2_FINAL")

    expected = 1 + 3*n + s_qubits
    if qc.num_qubits != expected:
        raise AssertionError(f"unexpected quantum width: got {qc.num_qubits}, expected {expected}")
    return qc


def build_report(n: int = 256, p: int = SECP256K1_P, *, s_qubits: Optional[int] = None) -> dict:
    if s_qubits is None: s_qubits = shared_eea_s_qubits(n)
    qc = build_point_addition_fig14_quadratic(n=n, p=p, s_qubits=s_qubits)
    budget = WrappedPointAdditionBudget(n=n, s_qubits=s_qubits)
    return {
        "budget": budget.as_dict(),
        "shared_eea_layout": shared_eea_layout(n).as_dict(),
        "qiskit_num_qubits": int(qc.num_qubits),
        "qiskit_num_clbits": int(qc.num_clbits),
        "qregs": {reg.name: int(reg.size) for reg in qc.qregs},
        "cregs": {reg.name: int(reg.size) for reg in qc.cregs},
        "has_extra_E_or_R_registers": any(reg.name.startswith(("E_", "R_")) for reg in qc.qregs),
        "top_level_ops": {str(k): int(v) for k, v in qc.count_ops().items()},
        "arithmetic_backend": "quadratic: Gidney measurement-vented constant add/compare + Cuccaro-style quantum add/compare; FASTDUAL low-aux EEA with Algorithm-1 wrapper",
        "uses_definitionless_placeholders": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--p", type=lambda s: int(s, 0), default=SECP256K1_P)
    ap.add_argument("--x2", type=lambda s: int(s, 0), default=0)
    ap.add_argument("--y2", type=lambda s: int(s, 0), default=0)
    ap.add_argument("--s-qubits", type=int, default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    report = build_report(args.n, args.p, s_qubits=args.s_qubits)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.json:
        Path(args.json).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
