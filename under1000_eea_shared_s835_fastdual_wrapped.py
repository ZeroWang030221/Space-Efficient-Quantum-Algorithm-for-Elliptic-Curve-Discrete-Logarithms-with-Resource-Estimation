from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Optional, Sequence

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Instruction, Qubit

import eea_circuit_s835_fastdual as eea
from quadratic_lazy_instruction import LazyDefinedInstruction
from quadratic_gidney_arithmetic import (
    append_gidney_compare_ge_const,
    append_gidney_add_const_mod2n,
)
from under1000_modular_arithmetic_base import _append_cx_multi

SECP256K1_P = (1 << 256) - (1 << 32) - 977


@dataclass(frozen=True)
class SharedEEALayout:
    n: int
    len_width: int
    shift_width: int
    T_max: int
    work2_tail: int = 3
    work1_tail: int = 3
    persistent_controls: int = 4  # Phase1, Phase2, Iter, Sign. Ctrl lives in Aux[0].
    step_aux: int = 20            # includes temporary Ctrl in Aux[0].

    @property
    def length_registers(self) -> int:
        return 3 * self.len_width + self.shift_width

    @property
    def s_qubits(self) -> int:
        return self.work2_tail + self.work1_tail + self.persistent_controls + self.length_registers + self.step_aux

    @property
    def forward_gate_qubits(self) -> int:
        return 2 * self.n + self.s_qubits

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update({
            "controls": self.persistent_controls,
            "length_registers": self.length_registers,
            "s_qubits": self.s_qubits,
            "forward_gate_qubits": self.forward_gate_qubits,
            "point_addition_quantum_qubits_with_control": 1 + 3 * self.n + self.s_qubits,
        })
        return d


def shared_eea_layout(n: int, *, T_max: Optional[int] = None) -> SharedEEALayout:
    cfg = eea.get_n_config(n)
    lw = int(cfg["len_width"]); sw = int(cfg["shift_width"])
    if T_max is None:
        T_max = int(cfg["T_max"])
    aux = int(eea.qiskit_paper_aux_size(n, lw, sw, T_max, include_algorithm1=False))
    return SharedEEALayout(n=int(n), len_width=lw, shift_width=sw, T_max=int(T_max), step_aux=aux)


def shared_eea_s_qubits(n: int, *, T_max: Optional[int] = None) -> int:
    return shared_eea_layout(n, T_max=T_max).s_qubits


def split_shared_s(S: Sequence[Qubit], n: int, *, T_max: Optional[int] = None) -> dict[str, list[Qubit]]:
    layout = shared_eea_layout(n, T_max=T_max)
    S = list(S)
    if len(S) < layout.s_qubits:
        raise ValueError(f"shared S too small: need {layout.s_qubits}, got {len(S)}")
    off = 0; out: dict[str, list[Qubit]] = {}
    out["work2_tail"] = S[off:off+3]; off += 3
    out["work1_tail"] = S[off:off+3]; off += 3
    for name in ["Phase1", "Phase2", "Iter", "Sign"]:
        out[name] = S[off:off+1]; off += 1
    lw = layout.len_width; sw = layout.shift_width
    out["l_t"] = S[off:off+lw]; off += lw
    out["l_q"] = S[off:off+lw]; off += lw
    out["l_s"] = S[off:off+sw]; off += sw
    out["l_rp"] = S[off:off+lw]; off += lw
    out["Aux"] = S[off:off+layout.step_aux]; off += layout.step_aux
    out["unused"] = S[off:]
    return out


def _apply_source_to_target_permutation(qc: QuantumCircuit, wires: Sequence[Qubit], target_of_source: Sequence[int]) -> None:
    wires = list(wires); m = len(wires)
    if sorted(target_of_source) != list(range(m)):
        raise ValueError("target_of_source is not a permutation")
    desired_at_pos = [None] * m
    for source, target in enumerate(target_of_source):
        desired_at_pos[target] = source
    current_at_pos = list(range(m))
    for pos in range(m):
        want = desired_at_pos[pos]
        if current_at_pos[pos] == want:
            continue
        j = current_at_pos.index(want)
        qc.swap(wires[pos], wires[j])
        current_at_pos[pos], current_at_pos[j] = current_at_pos[j], current_at_pos[pos]


def _prepare_work2_from_little_endian_x(qc: QuantumCircuit, X: Sequence[Qubit], tail: Sequence[Qubit]) -> list[Qubit]:
    """Permute little-endian X into the paper's Work2=[000,x]_big_endian layout."""
    X = list(X); tail = list(tail); n = len(X)
    work2 = X + tail
    target = [0] * (n + 3)
    for i in range(n):
        target[i] = 3 + (n - 1 - i)
    target[n] = 0; target[n+1] = 1; target[n+2] = 2
    _apply_source_to_target_permutation(qc, work2, target)
    return work2


def _set_big_endian_constant(qc: QuantumCircuit, reg_be: Sequence[Qubit], value: int) -> None:
    width = len(reg_be)
    for i, q in enumerate(reg_be):
        if (int(value) >> (width - 1 - i)) & 1:
            qc.x(q)


def _toggle_work1_constant(qc: QuantumCircuit, Work1: Sequence[Qubit], p: int) -> None:
    Work1 = list(Work1)
    qc.x(Work1[0])
    _set_big_endian_constant(qc, Work1[3:], p)


def _xor_const_into_reg(qc: QuantumCircuit, reg: Sequence[Qubit], value: int) -> None:
    mask = (1 << len(reg)) - 1
    value &= mask
    for i, q in enumerate(reg):
        if (value >> i) & 1:
            qc.x(q)


def _append_controlled_const_minus_mod2n_gidney(
    qc: QuantumCircuit,
    ctrl: Qubit,
    reg_le: Sequence[Qubit],
    const: int,
    dirty: Sequence[Qubit],
    clean3: Sequence[Qubit],
    cbits: Sequence,
) -> None:
    """If ctrl=1, reg <- const - reg (mod 2^n), restoring dirty/clean.

    For ctrl=1, bitwise complement + add 1 + add const gives const-reg modulo
    2^n. For ctrl=0 every suboperation is identity. Constant additions use the
    Gidney measurement-vented backend, so this wrapper stays linear in n.
    """
    reg = list(reg_le)
    for q in reg:
        qc.cx(ctrl, q)
    append_gidney_add_const_mod2n(qc, reg, 1, dirty[:max(0, len(reg)-1)], clean3, cbits, ctrl=ctrl)
    append_gidney_add_const_mod2n(qc, reg, const, dirty[:max(0, len(reg)-1)], clean3, cbits, ctrl=ctrl)


def _xor_encoded_bit_length_big_endian(qc: QuantumCircuit, bits_be: Sequence[Qubit], target_len: Sequence[Qubit], flag: Qubit) -> None:
    """XOR len(bits_be)-1 into target_len under the paper's encoded-length convention.

    This is a real reversible circuit. It scans for the first one in the
    big-endian string. The all-zero case writes encoded zero length, i.e. all
    ones. The temporary flag is restored after each case.
    """
    bits = list(bits_be); n = len(bits); mask = (1 << len(target_len)) - 1
    for first_one in range(n):
        encoded = (n - first_one - 1) & mask
        for j in range(first_one):
            qc.x(bits[j])
        _append_cx_multi(qc, bits[:first_one+1], flag)
        for i, q in enumerate(target_len):
            if (encoded >> i) & 1:
                qc.cx(flag, q)
        _append_cx_multi(qc, bits[:first_one+1], flag)
        for j in reversed(range(first_one)):
            qc.x(bits[j])
    # zero input -> encoded length 0 is -1 = all ones
    for q in bits:
        qc.x(q)
    _append_cx_multi(qc, bits, flag)
    for q in target_len:
        qc.cx(flag, q)
    _append_cx_multi(qc, bits, flag)
    for q in reversed(bits):
        qc.x(q)


@lru_cache(maxsize=None)
def _algorithm3_step_fastdual_gate(n: int, len_width: int, shift_width: int, T_max: int, aux_size: int, T: int) -> Instruction:
    work_size = n + 3
    num_qubits = 4 + 2 * work_size + 3 * len_width + shift_width + aux_size

    def _builder() -> QuantumCircuit:
        eea.set_measurement_uncompute(True)
        Phase1 = QuantumRegister(1, "Phase1")
        Phase2 = QuantumRegister(1, "Phase2")
        Iter = QuantumRegister(1, "Iter")
        Sign = QuantumRegister(1, "Sign")
        Work1 = QuantumRegister(work_size, "Work1")
        Work2 = QuantumRegister(work_size, "Work2")
        l_t = QuantumRegister(len_width, "l_t")
        l_q = QuantumRegister(len_width, "l_q")
        l_s = QuantumRegister(shift_width, "l_s")
        l_rp = QuantumRegister(len_width, "l_rp")
        Aux = QuantumRegister(aux_size, "Aux")
        q = QuantumCircuit(Phase1, Phase2, Iter, Sign, Work1, Work2, l_t, l_q, l_s, l_rp, Aux,
                           name=f"ALG3_STEP_FASTDUAL_WRAPPED_DEF_T{T}_{n}")
        eea.append_one_step_T(q, T=T, n=n, len_width=len_width, shift_width=shift_width,
                              Phase1=Phase1, Phase2=Phase2, Iter=Iter, Sign=Sign,
                              Work1=Work1, Work2=Work2, l_t=l_t, l_q=l_q, l_s=l_s, l_rp=l_rp, Aux=Aux)
        return q

    return LazyDefinedInstruction(f"ALG3_STEP_FASTDUAL_WRAPPED_T{T}_{n}", num_qubits, 0, _builder)


@lru_cache(maxsize=None)
def forward_eea_shared_definition(n: int, p: int = SECP256K1_P, T_max: Optional[int] = None) -> QuantumCircuit:
    layout = shared_eea_layout(n, T_max=T_max)
    X = QuantumRegister(n, "X_le")
    A = QuantumRegister(n, "A_large_workspace")
    S = QuantumRegister(layout.s_qubits, "S_shared")
    m = ClassicalRegister(max(1, n), "m_eea_wrapper")
    qc = QuantumCircuit(X, A, S, m, name=f"EEA_SHARED_ALG3_FASTDUAL_WRAPPED_DEF_{n}")
    parts = split_shared_s(S, n, T_max=T_max)

    Work2 = _prepare_work2_from_little_endian_x(qc, X, parts["work2_tail"])
    Work1 = parts["work1_tail"] + list(A)
    _toggle_work1_constant(qc, Work1, p)

    work2_rprime_le = list(reversed(Work2[3:3+n]))
    dirty = Work1[:n]
    clean3 = [parts["Aux"][1], parts["Aux"][2], parts["Aux"][3]]
    cbits = list(m)[:n]

    # Algorithm 1 preprocessing: if x > p/2, set Iter and run EEA on p-x.
    append_gidney_compare_ge_const(qc, work2_rprime_le, p // 2 + 1, parts["Iter"][0], dirty, clean3, cbits)
    _append_controlled_const_minus_mod2n_gidney(qc, parts["Iter"][0], work2_rprime_le, p, dirty, clean3, cbits)

    lw = layout.len_width; sw = layout.shift_width
    _xor_const_into_reg(qc, parts["l_q"], (1 << lw) - 1)       # ell_q = 0 encoded as -1
    _xor_const_into_reg(qc, parts["l_s"], (1 << sw) - 1)       # ell_s = 0 encoded as -1
    _xor_const_into_reg(qc, parts["l_t"], 0)                   # ell_t = 1 encoded as 0
    _xor_encoded_bit_length_big_endian(qc, Work2[3:3+n], parts["l_rp"], parts["Aux"][0])

    step_qubits = [parts["Phase1"][0], parts["Phase2"][0], parts["Iter"][0], parts["Sign"][0],
                  *Work1, *Work2, *parts["l_t"], *parts["l_q"], *parts["l_s"], *parts["l_rp"], *parts["Aux"]]
    for T in range(1, layout.T_max + 1):
        qc.append(_algorithm3_step_fastdual_gate(n, lw, sw, layout.T_max, layout.step_aux, T), step_qubits)

    # Algorithm 1 postprocessing: if Iter=0, convert t' to the positive inverse p-t'.
    qc.x(parts["Iter"][0])
    _append_controlled_const_minus_mod2n_gidney(qc, parts["Iter"][0], Work2[:n], p, dirty, clean3, cbits)
    qc.x(parts["Iter"][0])
    return qc


def eea_forward_shared_instruction(n: int, p: int = SECP256K1_P, *, T_max: Optional[int] = None, lazy_definition: bool = True) -> Instruction:
    layout = shared_eea_layout(n, T_max=T_max)
    builder = lambda: forward_eea_shared_definition(n, p, T_max)
    if lazy_definition:
        return LazyDefinedInstruction(f"EEA_FORWARD_SHARED_ALG3_FASTDUAL_WRAPPED_{n}", 2*n + layout.s_qubits, max(1, n), builder)
    q = builder()
    return q.to_instruction(label=f"EEA_FORWARD_SHARED_ALG3_FASTDUAL_WRAPPED_{n}")


def eea_inverse_shared_instruction(n: int, p: int = SECP256K1_P, *, T_max: Optional[int] = None, lazy_definition: bool = True) -> Instruction:
    # Dynamic measurement-based step definitions cannot be inverted by Qiskit's
    # unitary inverse.  Fig.15 uses the reverse logical block with the same
    # resource count; expose it as a separate definition-carrying instruction.
    layout = shared_eea_layout(n, T_max=T_max)
    builder = lambda: forward_eea_shared_definition(n, p, T_max)
    if lazy_definition:
        return LazyDefinedInstruction(f"EEA_FORWARD_SHARED_ALG3_FASTDUAL_WRAPPED_{n}_dg", 2*n + layout.s_qubits, max(1, n), builder)
    q = builder()
    return q.to_instruction(label=f"EEA_FORWARD_SHARED_ALG3_FASTDUAL_WRAPPED_{n}_dg")


# Compatibility aliases used by some older scripts.
eea_forward_shared_gate = eea_forward_shared_instruction
eea_inverse_shared_gate = eea_inverse_shared_instruction


def width_report(n: int = 256, *, T_max: Optional[int] = None) -> dict:
    layout = shared_eea_layout(n, T_max=T_max)
    return {"shared_eea_layout": layout.as_dict(), "point_addition_quantum_qubits": 1 + 3*n + layout.s_qubits}


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=256)
    args = ap.parse_args()
    print(json.dumps(width_report(args.n), indent=2, sort_keys=True))
