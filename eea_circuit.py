from typing import Sequence, List, Literal, Optional
from pathlib import Path
from collections import Counter
from functools import lru_cache
from qiskit.circuit import QuantumCircuit, QuantumRegister, Gate, Instruction
from qiskit import transpile
from qiskit.qasm2 import dumps as qasm2_dumps

import math
import time
import argparse
import gc

_phi = (math.sqrt(5.0) + 1.0) / 2.0
C_EEA = 1.0 / math.log2(_phi)

N_CONFIG = {
    64: {"len_width": 7, "T_max": 371},
    128: {"len_width": 8, "T_max": 743},
    160: {"len_width": 9, "T_max": 927},
    192: {"len_width": 9, "T_max": 1111},
    224: {"len_width": 9, "T_max": 1295},
    256: {"len_width": 9, "T_max": 1479},
    384: {"len_width": 10, "T_max": 2215},
    512: {"len_width": 10, "T_max": 2955},
}

PRIMITIVE_OPS = {"ccx", "cx", "x"}
_INST_COUNT_CACHE = {}


def _toggle_x_for_constant(qc: QuantumCircuit, reg: Sequence, value: int) -> None:
    n = len(reg)
    mask = value % (1 << n)
    for i in range(n):
        if (mask >> i) & 1:
            qc.x(reg[i])


def add_const_mod_2n(qc: QuantumCircuit, reg: Sequence, value: int, scratch: Sequence) -> None:
    n = len(reg)
    if len(scratch) < n + 2:
        raise ValueError(f"add_const_mod_2n needs >= {n + 2} scratch qubits, got {len(scratch)}")
    const = scratch[:n]
    carry = scratch[n]
    z = scratch[n + 1]
    _toggle_x_for_constant(qc, const, value)
    qc.append(cuccaro_add_mod_2n_gate(n), list(const) + list(reg) + [carry] + [z])
    _toggle_x_for_constant(qc, const, value)


def add_const_mod_2n_gate(n: int, value: int, name: str | None = None) -> Gate:
    if name is None:
        name = f"ADD_CONST_{value}_MOD_2^{n}"

    const = QuantumRegister(n, "const")
    reg = QuantumRegister(n, "reg")
    carry = QuantumRegister(1, "carry")
    z = QuantumRegister(1, "z")

    qc = QuantumCircuit(const, reg, carry, z, name=name)

    _toggle_x_for_constant(qc, const, value)
    qc.append(cuccaro_add_mod_2n_gate(n), list(const) + list(reg) + [carry[0]] + [z[0]])

    _toggle_x_for_constant(qc, const, value)

    return qc.to_gate()


def sub_const_mod_2n(qc: QuantumCircuit, reg: Sequence, value: int, scratch: Sequence) -> None:
    n = len(reg)
    if len(scratch) < n + 2:
        raise ValueError(f"sub_const_mod_2n needs >= {n + 2} scratch qubits, got {len(scratch)}")
    const = scratch[:n]
    carry = scratch[n]
    z = scratch[n + 1]
    _toggle_x_for_constant(qc, const, value)
    qc.append(cuccaro_add_mod_2n_gate(n).inverse(), list(const) + list(reg) + [carry] + [z])
    _toggle_x_for_constant(qc, const, value)


def sub_const_mod_2n_gate(n: int, value: int, name: str | None = None) -> Gate:
    if name is None:
        name = f"SUB_CONST_{value}_MOD_2^{n}"

    const = QuantumRegister(n, "const")
    reg = QuantumRegister(n, "reg")
    carry = QuantumRegister(1, "carry")
    z = QuantumRegister(1, "z")
    qc = QuantumCircuit(const, reg, carry, z, name=name)

    _toggle_x_for_constant(qc, const, value)
    qc.append(cuccaro_add_mod_2n_gate(n).inverse(), list(const) + list(reg) + [carry[0]] + [z[0]])
    _toggle_x_for_constant(qc, const, value)

    return qc.to_gate()


def _maj(qc: QuantumCircuit, a, b, c) -> None:
    qc.cx(a, b)
    qc.cx(a, c)
    qc.ccx(c, b, a)


def _uma(qc: QuantumCircuit, a, b, c) -> None:
    qc.ccx(c, b, a)
    qc.cx(a, c)
    qc.cx(c, b)


def _maj_inv(qc: QuantumCircuit, a, b, c) -> None:
    qc.ccx(c, b, a)
    qc.cx(a, c)
    qc.cx(a, b)


def _uma_inv(qc: QuantumCircuit, a, b, c) -> None:
    qc.cx(c, b)
    qc.cx(a, c)
    qc.ccx(c, b, a)


def cuccaro_add_mod_2n_gate(n: int, name: str = "ADD_Z") -> Gate:
    a = QuantumRegister(n, "a")
    b = QuantumRegister(n, "b")
    c0 = QuantumRegister(1, "c")
    z = QuantumRegister(1, "z")
    qc = QuantumCircuit(a, b, c0, z, name=name)
    c = c0[0]

    _maj(qc, a[0], b[0], c)
    for i in range(1, n):
        _maj(qc, a[i], b[i], a[i - 1])

    qc.cx(a[n - 1], z[0])

    for i in reversed(range(1, n)):
        _uma(qc, a[i], b[i], a[i - 1])
    _uma(qc, a[0], b[0], c)

    return qc.to_gate()


def cuccaro_sub_mod_2n_gate(n: int, name: str = "SUB") -> Gate:
    return cuccaro_add_mod_2n_gate(n, name=name).inverse()


def len_adder(len_width: int) -> Gate:
    return cuccaro_add_mod_2n_gate(len_width, name="len_adder")


def len_sub(len_width: int) -> Gate:
    return len_adder(len_width).inverse()


def constant_sub(len_width: int, K_star: int, len_sub_gate: Gate) -> Gate:
    _c_a = QuantumRegister(len_width, "const")
    _c_b = QuantumRegister(len_width, "l_q")
    _c_c = QuantumRegister(1, "c")
    _c_z = QuantumRegister(1, "z")
    _qc = QuantumCircuit(_c_a, _c_b, _c_c, _c_z, name=f"constant_sub_{K_star}")

    _toggle_x_for_constant(_qc, _c_a, K_star)
    _qc.append(len_sub_gate, list(_c_a) + list(_c_b) + [_c_c[0]] + [_c_z[0]])
    _toggle_x_for_constant(_qc, _c_a, K_star)

    return _qc.to_gate()


def constant_adder(len_width: int, k_star: int, len_adder_gate: Gate) -> Gate:
    _a_a = QuantumRegister(len_width, "const")
    _a_b = QuantumRegister(len_width, "l_q")
    _a_c = QuantumRegister(1, "c")
    _a_z = QuantumRegister(1, "z")
    _qc = QuantumCircuit(_a_a, _a_b, _a_c, _a_z, name=f"constant_adder_{k_star}")

    _toggle_x_for_constant(_qc, _a_a, k_star)
    _qc.append(len_adder_gate, list(_a_a) + list(_a_b) + [_a_c[0]] + [_a_z[0]])
    _toggle_x_for_constant(_qc, _a_a, k_star)

    return _qc.to_gate()


def cuccaro_incrementer_gate(n: int, name: str = "INC_Z") -> Gate:
    a = QuantumRegister(n, "a")
    b = QuantumRegister(n, "b")
    c0 = QuantumRegister(1, "c")
    z = QuantumRegister(1, "z")
    qc = QuantumCircuit(a, b, c0, z, name=name)

    _maj(qc, a[0], b[0], c0[0])

    for i in range(1, n):
        qc.ccx(a[i - 1], b[i], a[i])

    qc.cx(a[n - 1], z[0])

    for i in reversed(range(1, n)):
        qc.ccx(a[i - 1], b[i], a[i])
        qc.cx(a[i - 1], b[i])

    _uma(qc, a[0], b[0], c0[0])

    return qc.to_gate()


def constant_add1(len_width: int, len_adder_gate: Gate) -> Gate:
    inc_gate = cuccaro_incrementer_gate(len_width, name="constant_add_1_opt")
    _o_a = QuantumRegister(len_width, "const")
    _o_b = QuantumRegister(len_width, "l_q")
    _o_c = QuantumRegister(1, "c")
    _o_z = QuantumRegister(1, "z")
    _qc = QuantumCircuit(_o_a, _o_b, _o_c, _o_z, name="constant_add_1")

    _qc.append(inc_gate, list(_o_a) + list(_o_b) + [_o_c[0]] + [_o_z[0]])
    return _qc.to_gate()


def mcx_vchain(qc, ctrls: Sequence, target, ancillas: Sequence):
    m = len(ctrls)
    if m == 0:
        qc.x(target)
        return
    if m == 1:
        qc.cx(ctrls[0], target)
        return
    if m == 2:
        qc.ccx(ctrls[0], ctrls[1], target)
        return

    need = m - 2
    if len(ancillas) < need:
        raise ValueError(f"mcx_vchain needs >= {need} ancillas, got {len(ancillas)}")

    qc.ccx(ctrls[0], ctrls[1], ancillas[0])
    for i in range(2, m - 1):
        qc.ccx(ctrls[i], ancillas[i - 2], ancillas[i - 1])

    qc.ccx(ctrls[m - 1], ancillas[m - 3], target)

    for i in range(m - 2, 1, -1):
        qc.ccx(ctrls[i], ancillas[i - 2], ancillas[i - 1])
    qc.ccx(ctrls[0], ctrls[1], ancillas[0])


def controlled_add_one(
        qc: QuantumCircuit,
        reg: QuantumRegister,
        ctrls: List,
        scratch: List,
) -> None:
    n = len(reg)
    for i in range(n - 1, 0, -1):
        mcx_vchain(qc, ctrls + list(reg[:i]), reg[i], scratch)
    mcx_vchain(qc, ctrls, reg[0], scratch)


def controlled_sub_one(
        qc: QuantumCircuit,
        reg: QuantumRegister,
        ctrls: List,
        scratch: List,
) -> None:
    n = len(reg)
    mcx_vchain(qc, ctrls, reg[0], scratch)
    for i in range(1, n):
        mcx_vchain(qc, ctrls + list(reg[:i]), reg[i], scratch)


def cswap_toffoli(qc, ctrl, a, b):
    qc.cx(b, a)
    qc.ccx(ctrl, a, b)
    qc.cx(b, a)


def inc_mod2n_uncontrolled(qc, reg, anc):
    n = len(reg)
    if n == 0:
        return
    if n == 1:
        qc.x(reg[0])
        return
    if len(anc) < n - 1:
        raise ValueError(f"inc_mod2n_uncontrolled needs {n - 1} ancillas, got {len(anc)}")
    c = anc

    qc.cx(reg[0], c[0])

    for i in range(1, n - 1):
        qc.ccx(reg[i], c[i - 1], c[i])

    qc.cx(c[n - 2], reg[n - 1])

    for i in range(n - 2, 0, -1):
        qc.ccx(reg[i], c[i - 1], c[i])
        qc.cx(c[i - 1], reg[i])

    qc.cx(reg[0], c[0])

    qc.x(reg[0])


def dec_mod2n_uncontrolled(qc, reg, anc):
    n = len(reg)
    if n == 0:
        return
    if n == 1:
        qc.x(reg[0])
        return
    if len(anc) < n - 1:
        raise ValueError(f"dec_mod2n_uncontrolled needs {n - 1} ancillas, got {len(anc)}")
    b = anc

    qc.x(reg[0])
    qc.cx(reg[0], b[0])
    qc.x(reg[0])

    for i in range(1, n - 1):
        qc.x(reg[i])
        qc.ccx(reg[i], b[i - 1], b[i])
        qc.x(reg[i])

    qc.cx(b[n - 2], reg[n - 1])

    for i in range(n - 2, 0, -1):
        qc.x(reg[i])
        qc.ccx(reg[i], b[i - 1], b[i])
        qc.x(reg[i])
        qc.cx(b[i - 1], reg[i])

    qc.x(reg[0])
    qc.cx(reg[0], b[0])
    qc.x(reg[0])

    qc.x(reg[0])


def ctrl_and_sign_cswap(
        qc: QuantumCircuit,
        ctrl_qubit,
        signbit,
        carry_qubit,
        work_qubit,
        sign_qubit,
        sign_control_value: int,
) -> None:
    if sign_control_value not in (0, 1):
        raise ValueError("sign_control_value must be 0 or 1.")

    if sign_control_value == 0:
        qc.x(signbit)

    qc.ccx(ctrl_qubit, signbit, carry_qubit)
    cswap_toffoli(qc, carry_qubit, work_qubit, sign_qubit)
    qc.ccx(ctrl_qubit, signbit, carry_qubit)

    if sign_control_value == 0:
        qc.x(signbit)


def inc_mod2n_1ctrl(qc, ctrl, reg: Sequence, anc: Sequence):
    n = len(reg)
    if n == 0:
        return
    if n == 1:
        qc.cx(ctrl, reg[0])
        return
    if len(anc) < n - 1:
        raise ValueError(f"inc_mod2n_1ctrl needs >= {n - 1} ancillas, got {len(anc)}")

    c = anc

    qc.ccx(ctrl, reg[0], c[0])
    qc.cx(ctrl, reg[0])

    for i in range(1, n - 1):
        qc.ccx(reg[i], c[i - 1], c[i])

    qc.cx(c[n - 2], reg[n - 1])

    for i in range(n - 2, 0, -1):
        qc.ccx(reg[i], c[i - 1], c[i])
        qc.cx(c[i - 1], reg[i])

    qc.cx(ctrl, reg[0])
    qc.ccx(ctrl, reg[0], c[0])
    qc.cx(ctrl, reg[0])


def dec_mod2n_1ctrl(qc, ctrl, reg: Sequence, anc: Sequence):
    n = len(reg)
    if n == 0:
        return
    if n == 1:
        qc.cx(ctrl, reg[0])
        return
    if len(anc) < n - 1:
        raise ValueError(f"dec_mod2n_1ctrl needs >= {n - 1} ancillas, got {len(anc)}")

    b = anc

    qc.x(reg[0])
    qc.ccx(ctrl, reg[0], b[0])
    qc.x(reg[0])

    qc.cx(ctrl, reg[0])

    for i in range(1, n - 1):
        qc.x(reg[i])
        qc.ccx(reg[i], b[i - 1], b[i])
        qc.x(reg[i])

    qc.cx(b[n - 2], reg[n - 1])

    for i in range(n - 2, 0, -1):
        qc.x(reg[i])
        qc.ccx(reg[i], b[i - 1], b[i])
        qc.x(reg[i])
        qc.cx(b[i - 1], reg[i])

    qc.cx(ctrl, reg[0])
    qc.x(reg[0])
    qc.ccx(ctrl, reg[0], b[0])
    qc.x(reg[0])
    qc.cx(ctrl, reg[0])


def _ccx_1ctrl(qc, ctrl, a, b, t, pool):
    mcx_vchain(qc, [ctrl, a, b], t, pool)


def _cx_1ctrl(qc, ctrl, c, t, pool):
    qc.ccx(ctrl, c, t)


def cuccaro_add_mod_2n_no_z_1ctrl_gate(n: int, name: str = "C_ADD_MOD_2N_NO_Z") -> Gate:
    Ctrl = QuantumRegister(1, "Ctrl")
    a = QuantumRegister(n, "a")
    b = QuantumRegister(n, "b")
    c0 = QuantumRegister(1, "c0")
    pool = QuantumRegister(max(1, n), "pool")
    qc = QuantumCircuit(Ctrl, a, b, c0, pool, name=name)

    ctrl = Ctrl[0]
    c = c0[0]

    _cx_1ctrl(qc, ctrl, c, b[0], pool)
    _cx_1ctrl(qc, ctrl, c, a[0], pool)
    _ccx_1ctrl(qc, ctrl, a[0], b[0], c, pool)

    for i in range(1, n):
        _cx_1ctrl(qc, ctrl, a[i - 1], b[i], pool)
        _cx_1ctrl(qc, ctrl, a[i - 1], a[i], pool)
        _ccx_1ctrl(qc, ctrl, a[i], b[i], a[i - 1], pool)

    for i in reversed(range(1, n)):
        _ccx_1ctrl(qc, ctrl, a[i], b[i], a[i - 1], pool)
        _cx_1ctrl(qc, ctrl, a[i - 1], a[i], pool)
        _cx_1ctrl(qc, ctrl, a[i], b[i], pool)

    _ccx_1ctrl(qc, ctrl, a[0], b[0], c, pool)
    _cx_1ctrl(qc, ctrl, c, a[0], pool)
    _cx_1ctrl(qc, ctrl, a[0], b[0], pool)

    return qc.to_gate()


def cuccaro_add_mod_2n_no_z_gate(n: int, name: str = "ADD_MOD_2N_NO_Z"):
    a = QuantumRegister(n, "a")
    b = QuantumRegister(n, "b")
    c0 = QuantumRegister(1, "c0")
    qc = QuantumCircuit(a, b, c0, name=name)

    c = c0[0]
    _maj(qc, a[0], b[0], c)
    for i in range(1, n):
        _maj(qc, a[i], b[i], a[i - 1])

    for i in reversed(range(1, n)):
        _uma(qc, a[i], b[i], a[i - 1])
    _uma(qc, a[0], b[0], c)

    return qc.to_gate()


def len_adder_gate(len_width: int, name: str = "LEN_ADD"):
    return cuccaro_add_mod_2n_gate(len_width, name=name)


def len_sub_gate(len_width: int, name: str = "LEN_SUB"):
    return len_adder_gate(len_width, name=name).inverse()


def controlled_add_one_1ctrl(qc: QuantumCircuit, ctrl, reg, scratch):
    n = len(reg)
    if n < 1:
        return
    if n >= 3 and len(scratch) < (n - 2):
        raise ValueError(f"controlled_add_one needs >= {n - 2} scratch ancillas, got {len(scratch)}")

    qc.cx(ctrl, reg[0])

    for i in range(1, n):
        ctrls = [ctrl] + [reg[j] for j in range(i)]
        mcx_vchain(qc, ctrls, reg[i], scratch)


def pre_shift_gate(*, work_size: int, len_width: int, name: str = "PRE_SHIFT") -> Gate:
    Phase1 = QuantumRegister(1, "Phase1")
    Phase2 = QuantumRegister(1, "Phase2")
    Work2 = QuantumRegister(work_size, "Work2")
    l_s = QuantumRegister(len_width, "l_s")
    ZeroS = QuantumRegister(len_width + 4, "ZeroS")
    qc = QuantumCircuit(Phase1, Phase2, Work2, l_s, ZeroS, name=name)

    n = len_width
    phase1_is0 = ZeroS[0]
    both = ZeroS[1]
    const = list(ZeroS[2:2 + n])
    carry = ZeroS[2 + n]
    z = ZeroS[2 + n + 1]

    qc.x(Phase1[0])
    qc.cx(Phase1[0], phase1_is0)
    qc.x(Phase1[0])

    for i in range(work_size - 1):
        cswap_toffoli(qc, phase1_is0, Work2[i], Work2[i + 1])

    carry_chain = list(ZeroS[2:2 + (n - 1)])
    inc_mod2n_1ctrl(qc, phase1_is0, list(l_s), carry_chain)

    qc.ccx(phase1_is0, Phase2[0], both)

    W = work_size
    for i in range(W - 3, -1, -1):
        cswap_toffoli(qc, both, Work2[i], Work2[i + 1])
        cswap_toffoli(qc, both, Work2[i + 1], Work2[i + 2])

    borrow_chain = list(ZeroS[2:2 + (n - 1)])
    dec_mod2n_1ctrl(qc, both, list(l_s), borrow_chain)
    dec_mod2n_1ctrl(qc, both, list(l_s), borrow_chain)

    qc.ccx(phase1_is0, Phase2[0], both)
    qc.x(Phase1[0])
    qc.cx(Phase1[0], phase1_is0)
    qc.x(Phase1[0])

    return qc.to_gate()


def location_controlled_add_gate(
        *,
        n: int,
        k: int,
        K: int,
        len_width: int,
        name: str = "LC_ADD",
) -> Gate:
    if k > K:
        raise ValueError(f"Need k <= K, got k={k}, K={K}.")
    if K > n + 3:
        raise ValueError(f"Need K <= n+3, got K={K}, n+3={n + 3}.")
    if len_width < 4:
        raise ValueError("len_width must be >= 4 for this implementation.")

    W = K - k + 1
    K1_star = K - 3
    K2_star = (n + 3) - K

    Phase1 = QuantumRegister(1, "Phase1")
    Phase2 = QuantumRegister(1, "Phase2")
    Zero = QuantumRegister(2 + len_width, "Zero")
    Sign = QuantumRegister(1, "Sign")
    Work1 = QuantumRegister(W, "Work1")
    Work2 = QuantumRegister(W, "Work2")
    l_t = QuantumRegister(len_width, "l_t")
    l_q = QuantumRegister(len_width, "l_q")
    l_s = QuantumRegister(len_width, "l_s")

    qc = QuantumCircuit(
        Phase1, Phase2,
        Zero, Sign,
        Work1, Work2,
        l_t, l_q, l_s,
        name=name
    )

    carry = Zero[0]
    z = Zero[1]
    scratch = list(Zero[2:])
    enable = scratch[0]
    tmp = scratch[1]
    cond = scratch[2]
    pool = scratch[3:]
    len_scratch = [tmp, cond] + pool

    sign_lq = l_q[len_width - 1]
    sign_ls = l_s[len_width - 1]

    len_adder_gate = len_adder(len_width)
    len_sub_gate = len_sub(len_width)
    sub_K1_gate = constant_sub(len_width, K1_star, len_sub_gate)
    add_K1_gate = constant_adder(len_width, K1_star, len_adder_gate)
    sub_K2_gate = constant_sub(len_width, K2_star, len_sub_gate)
    add_K2_gate = constant_adder(len_width, K2_star, len_adder_gate)

    qc.append(len_adder_gate, list(l_t) + list(l_q) + [carry] + [z])
    qc.append(sub_K1_gate, scratch + list(l_q) + [carry] + [z])
    qc.append(sub_K2_gate, scratch + list(l_s) + [carry] + [z])

    qc.ccx(Phase2[0], Sign[0], tmp)
    qc.x(tmp)
    qc.x(Phase1[0])
    qc.ccx(Phase1[0], tmp, enable)
    qc.x(Phase1[0])
    qc.x(tmp)
    qc.ccx(Phase2[0], Sign[0], tmp)

    u_le = list(reversed(list(Work1)))
    v_le = list(reversed(list(Work2)))

    for i in range(W):
        a = v_le[i]
        b = u_le[i]

        qc.ccx(enable, sign_ls, tmp)
        qc.ccx(tmp, sign_lq, cond)

        qc.ccx(cond, carry, b)
        qc.ccx(cond, carry, a)
        mcx_vchain(qc, [cond, a, b], carry, pool)

        qc.ccx(tmp, sign_lq, cond)
        qc.ccx(enable, sign_ls, tmp)

        if i < W - 1:
            dec_mod2n_uncontrolled(qc, l_s, len_scratch)

            inc_mod2n_uncontrolled(qc, l_q, len_scratch)

    for i in reversed(range(W)):
        a = v_le[i]
        b = u_le[i]

        qc.ccx(enable, sign_ls, tmp)
        qc.ccx(tmp, sign_lq, cond)

        mcx_vchain(qc, [cond, a, b], carry, pool)
        qc.ccx(cond, carry, a)
        qc.ccx(cond, a, b)

        qc.ccx(tmp, sign_lq, cond)
        qc.ccx(enable, sign_ls, tmp)

        if i > 0:
            inc_mod2n_uncontrolled(qc, l_s, len_scratch)

            dec_mod2n_uncontrolled(qc, l_q, len_scratch)

    qc.ccx(Phase2[0], Sign[0], tmp)
    qc.x(tmp)
    qc.x(Phase1[0])
    qc.ccx(Phase1[0], tmp, enable)
    qc.x(Phase1[0])
    qc.x(tmp)
    qc.ccx(Phase2[0], Sign[0], tmp)

    qc.append(add_K2_gate, scratch + list(l_s) + [carry] + [z])
    qc.append(add_K1_gate, scratch + list(l_q) + [carry] + [z])
    qc.append(len_sub_gate, list(l_t) + list(l_q) + [carry] + [z])

    return qc.to_gate()


def _location_controlled_sub_gate_impl(k: int, K: int, work_nbits: int, lt_bits: int, lq_bits: int, ls_bits: int,
                                       zero_bits: int, label: str = "LC_SUB") -> Gate:
    W = K - k + 1
    if W <= 0:
        raise ValueError("Need K >= k")

    K1_star = K - 3
    K2_star = work_nbits - K

    Ctrl = QuantumRegister(1, "Ctrl")
    Zero = QuantumRegister(zero_bits, "Zero")
    Sign = QuantumRegister(1, "Sign")
    Work1 = QuantumRegister(W, "Work1")
    Work2 = QuantumRegister(W, "Work2")
    lt = QuantumRegister(lt_bits, "l_t")
    lq = QuantumRegister(lq_bits, "l_q")
    ls = QuantumRegister(ls_bits, "l_s")

    qc = QuantumCircuit(Ctrl, Zero, Sign, Work1, Work2, lt, lq, ls, name=label)

    if zero_bits < 3:
        raise ValueError("zero_bits must be >= 3")
    carry = Zero[0]
    active = Zero[1]
    tmp = Zero[2]
    scratch_pool = list(Zero[3:])
    len_width = ls_bits
    len_chain = scratch_pool[:len_width - 1]

    if len(scratch_pool) < max(0, max(lt_bits, lq_bits, ls_bits) - 2):
        raise ValueError("Not enough Zero scratch qubits for length-reg ops (need ~max(len_reg)-2).")

    z = scratch_pool[0]
    pool_for_consts = scratch_pool[1:]

    if lt_bits != lq_bits:
        raise ValueError("This implementation assumes lt_bits == lq_bits (as in typical setups).")
    qc.append(cuccaro_add_mod_2n_gate(lt_bits, name="ADD_lt_into_lq"), [*lt, *lq, carry, z])

    sub_const_mod_2n(qc, lq, K1_star, pool_for_consts)

    sub_const_mod_2n(qc, ls, K2_star, pool_for_consts)

    lq_msb = lq[-1]
    ls_msb = ls[-1]

    for idx in range(W - 1, -1, -1):
        mcx_vchain(qc, [Ctrl[0], ls_msb, lq_msb], active, [tmp])

        qc.ccx(active, Work2[idx], Work1[idx])
        qc.ccx(active, carry, Work2[idx])
        mcx_vchain(qc, [active, Work2[idx], Work1[idx]], carry, [tmp])

        mcx_vchain(qc, [Ctrl[0], ls_msb, lq_msb], active, [tmp])

        if idx != 0:
            dec_mod2n_uncontrolled(qc, ls, len_chain)
            inc_mod2n_uncontrolled(qc, lq, len_chain)

    qc.ccx(Ctrl[0], carry, Sign[0])

    for idx in range(0, W):
        mcx_vchain(qc, [Ctrl[0], ls_msb, lq_msb], active, [tmp])

        mcx_vchain(qc, [active, Work2[idx], Work1[idx]], carry, [tmp])
        qc.ccx(active, carry, Work2[idx])
        qc.ccx(active, carry, Work1[idx])

        mcx_vchain(qc, [Ctrl[0], ls_msb, lq_msb], active, [tmp])

        if idx != W - 1:
            inc_mod2n_uncontrolled(qc, ls, len_chain)
            dec_mod2n_uncontrolled(qc, lq, len_chain)

    add_const_mod_2n(qc, ls, K2_star, pool_for_consts)
    add_const_mod_2n(qc, lq, K1_star, pool_for_consts)
    qc.append(cuccaro_sub_mod_2n_gate(lt_bits, name="SUB_lt_from_lq"), [*lt, *lq, carry, z])

    return qc.to_gate()


def location_controlled_sub_gate(*, n: int, k: int, K: int, len_width: int, label: str = "LC_SUB") -> Gate:
    work_nbits = n
    zero_bits = 6 + len_width
    return _location_controlled_sub_gate_impl(
        k=k, K=K, work_nbits=work_nbits,
        lt_bits=len_width, lq_bits=len_width, ls_bits=len_width,
        zero_bits=zero_bits, label=label
    )


def location_controlled_swap_gate(
        *,
        k: int,
        K: int,
        len_width: int,
        name: str = "LC_SWAP",
) -> Gate:
    if k > K:
        raise ValueError(f"Need k <= K, got k={k}, K={K}.")
    if len_width < 2:
        raise ValueError("len_width must be >= 2 (needs sign/MSB).")

    W = K - k + 1

    Phase1 = QuantumRegister(1, "Phase1")
    Phase2 = QuantumRegister(1, "Phase2")

    Zero = QuantumRegister(2 + len_width, "Zero")

    Sign = QuantumRegister(1, "Sign")
    Work1 = QuantumRegister(W, "Work1")
    l_t = QuantumRegister(len_width, "l_t")
    l_q = QuantumRegister(len_width, "l_q")

    qc = QuantumCircuit(Phase1, Phase2, Zero, Sign, Work1, l_t, l_q, name=name)

    carry = Zero[0]
    zflag = Zero[1]
    scratch = list(Zero[2:])
    signbit = l_q[len_width - 1]

    K_star = K - 3
    k_star = k - 2

    len_adder_gate = len_adder(len_width)
    len_sub_gate = len_sub(len_width)
    constant_sub_gate = constant_sub(len_width, K_star, len_sub_gate)
    constant_adder_gate = constant_adder(len_width, k_star, len_adder_gate)
    constant_add1_gate = constant_add1(len_width, len_adder_gate)

    qc.cx(Phase1[0], Phase2[0])

    arith_ops = []

    def _append_arith(g: Gate, wires: list) -> None:
        qc.append(g, wires)
        arith_ops.append((g, wires))

    _append_arith(len_adder_gate, list(l_t) + list(l_q) + [carry, zflag])

    _append_arith(constant_sub_gate, scratch + list(l_q) + [carry, zflag])

    ctrl_and_sign_cswap(qc, Phase2[0], signbit, carry, Work1[W - 1], Sign[0], sign_control_value=0)

    add1_times = 0
    if W >= 3:

        _toggle_x_for_constant(qc, scratch, 1)
        for idx in range(W - 2, 0, -1):
            ctrl_and_sign_cswap(qc, Phase2[0], signbit, carry, Work1[idx], Sign[0], sign_control_value=1)
            _append_arith(constant_add1_gate, scratch + list(l_q) + [carry, zflag])
            add1_times += 1
            ctrl_and_sign_cswap(qc, Phase2[0], signbit, carry, Work1[idx], Sign[0], sign_control_value=1)
        _toggle_x_for_constant(qc, scratch, 1)

    ctrl_and_sign_cswap(qc, Phase2[0], signbit, carry, Work1[0], Sign[0], sign_control_value=1)

    _append_arith(constant_adder_gate, scratch + list(l_q) + [carry, zflag])

    _append_arith(len_sub_gate, list(l_t) + list(l_q) + [carry, zflag])

    for g, wires in reversed(arith_ops):
        qc.append(g.inverse(), wires)

    controlled_sub_one(qc, l_q, [Phase2[0], Phase1[0]], scratch)

    qc.x(Phase1[0])
    controlled_add_one(qc, l_q, [Phase2[0], Phase1[0]], scratch)
    qc.x(Phase1[0])

    qc.cx(Phase1[0], Phase2[0])

    return qc.to_gate()


def location_controlled_add_gate_single(k: int,
                                        K: int,
                                        lt_bits: int,
                                        zero_bits: int,
                                        label: str = "LC_ADD_SINGLE") -> Gate:
    W = K - k + 1
    if W <= 0:
        raise ValueError("Need K >= k")

    Phase1 = QuantumRegister(1, "Phase1")
    Phase2 = QuantumRegister(1, "Phase2")
    Sign = QuantumRegister(1, "Sign")
    Zero = QuantumRegister(zero_bits, "Zero")
    Work1 = QuantumRegister(W, "Work1")
    Work2 = QuantumRegister(W, "Work2")
    lt = QuantumRegister(lt_bits, "l_t")

    qc = QuantumCircuit(Phase1, Phase2, Sign, Zero, Work1, Work2, lt, name=label)

    if zero_bits < 3:
        raise ValueError("zero_bits must be >= 3")
    carry = Zero[0]
    active = Zero[1]
    tmp = Zero[2]
    scratch_pool = list(Zero[3:])
    if len(scratch_pool) < max(0, lt_bits - 2):
        raise ValueError("Not enough Zero scratch qubits for l_t +/-1 ops (need ~lt_bits-2).")

    lt_msb = lt[-1]

    for idx in range(W - 1, -1, -1):
        qc.x(lt_msb)
        qc.ccx(Phase1[0], lt_msb, active)
        qc.x(lt_msb)

        qc.ccx(active, carry, Work2[idx])
        qc.ccx(active, carry, Work1[idx])
        mcx_vchain(qc, [active, Work1[idx], Work2[idx]], carry, [tmp])

        qc.x(lt_msb)
        qc.ccx(Phase1[0], lt_msb, active)
        qc.x(lt_msb)

        if idx != 0:
            sub_const_mod_2n(qc, lt, 1, scratch_pool)

    qc.ccx(Phase1[0], carry, Sign[0])

    for idx in range(0, W):
        qc.x(lt_msb)
        qc.ccx(Phase1[0], lt_msb, active)
        qc.x(lt_msb)

        mcx_vchain(qc, [active, Work1[idx], Work2[idx]], carry, [tmp])
        qc.ccx(active, carry, Work1[idx])
        qc.ccx(active, Work1[idx], Work2[idx])

        qc.x(lt_msb)
        qc.ccx(Phase1[0], lt_msb, active)
        qc.x(lt_msb)

        if idx != W - 1:
            add_const_mod_2n(qc, lt, 1, scratch_pool)

    return qc.to_gate()


def location_controlled_sub_gate_single(k: int,
                                        K: int,
                                        lt_bits: int,
                                        zero_bits: int,
                                        label: str = "LC_SUB_SINGLE") -> Gate:
    W = K - k + 1
    if W <= 0:
        raise ValueError("Need K >= k")

    Phase1 = QuantumRegister(1, "Phase1")
    Phase2 = QuantumRegister(1, "Phase2")
    Sign = QuantumRegister(1, "Sign")
    Zero = QuantumRegister(zero_bits, "Zero")
    Work1 = QuantumRegister(W, "Work1")
    Work2 = QuantumRegister(W, "Work2")
    lt = QuantumRegister(lt_bits, "l_t")

    qc = QuantumCircuit(Phase1, Phase2, Sign, Zero, Work1, Work2, lt, name=label)

    if zero_bits < 6:
        raise ValueError("zero_bits must be >= 6 for this gate (carry/active/tmp/cond/enable + scratch).")

    carry = Zero[0]
    active = Zero[1]
    tmp = Zero[2]
    cond_or = Zero[3]
    enable = Zero[4]
    scratch_pool = list(Zero[5:])
    if len(scratch_pool) < max(0, lt_bits - 2):
        raise ValueError("Not enough Zero scratch qubits for l_t +/-1 ops (need ~lt_bits-2).")

    lt_msb = lt[-1]

    qc.cx(Phase2[0], cond_or)
    qc.x(Sign[0])
    qc.cx(Sign[0], cond_or)
    qc.ccx(Phase2[0], Sign[0], cond_or)
    qc.x(Sign[0])

    qc.ccx(Phase1[0], cond_or, enable)

    for idx in range(W - 1, -1, -1):
        qc.x(lt_msb)
        qc.ccx(enable, lt_msb, active)
        qc.x(lt_msb)

        qc.ccx(active, Work1[idx], Work2[idx])
        qc.ccx(active, carry, Work1[idx])
        mcx_vchain(qc, [active, Work1[idx], Work2[idx]], carry, [tmp])

        qc.x(lt_msb)
        qc.ccx(enable, lt_msb, active)
        qc.x(lt_msb)

        if idx != 0:
            sub_const_mod_2n(qc, lt, 1, scratch_pool)

    for idx in range(0, W):
        qc.x(lt_msb)
        qc.ccx(enable, lt_msb, active)
        qc.x(lt_msb)

        mcx_vchain(qc, [active, Work1[idx], Work2[idx]], carry, [tmp])
        qc.ccx(active, carry, Work1[idx])
        qc.ccx(active, carry, Work2[idx])

        qc.x(lt_msb)
        qc.ccx(enable, lt_msb, active)
        qc.x(lt_msb)

        if idx != W - 1:
            add_const_mod_2n(qc, lt, 1, scratch_pool)

    qc.ccx(Phase1[0], cond_or, enable)

    qc.x(Sign[0])
    qc.ccx(Phase2[0], Sign[0], cond_or)
    qc.cx(Sign[0], cond_or)
    qc.x(Sign[0])
    qc.cx(Phase2[0], cond_or)

    return qc.to_gate()


def post_shift_gate(*, work_size: int, len_width: int, name: str = "POST_SHIFT") -> Gate:
    Phase1 = QuantumRegister(1, "Phase1")
    Phase2 = QuantumRegister(1, "Phase2")
    Work2 = QuantumRegister(work_size, "Work2")
    l_s = QuantumRegister(len_width, "l_s")
    ZeroS = QuantumRegister(len_width + 4, "ZeroS")
    qc = QuantumCircuit(Phase1, Phase2, Work2, l_s, ZeroS, name=name)

    n = len_width
    both = ZeroS[0]
    const = list(ZeroS[1:1 + n])
    carry = ZeroS[1 + n]
    z = ZeroS[1 + n + 1]

    for i in range(work_size - 1):
        cswap_toffoli(qc, Phase1[0], Work2[i], Work2[i + 1])

    carry_chain = list(ZeroS[1:1 + (n - 1)])
    inc_mod2n_1ctrl(qc, Phase1[0], list(l_s), carry_chain)

    qc.ccx(Phase1[0], Phase2[0], both)

    W = work_size
    for i in range(W - 3, -1, -1):
        cswap_toffoli(qc, both, Work2[i], Work2[i + 1])
        cswap_toffoli(qc, both, Work2[i + 1], Work2[i + 2])

    borrow_chain = list(ZeroS[1:1 + (n - 1)])
    dec_mod2n_1ctrl(qc, both, list(l_s), borrow_chain)
    dec_mod2n_1ctrl(qc, both, list(l_s), borrow_chain)

    qc.ccx(Phase1[0], Phase2[0], both)

    return qc.to_gate()


def phase_update_gate(*, len_width: int, name: str = "PHASE_UPDATE") -> Gate:
    Phase1 = QuantumRegister(1, "Phase1")
    Phase2 = QuantumRegister(1, "Phase2")
    Sign = QuantumRegister(1, "Sign")
    l_q = QuantumRegister(len_width, "l_q")
    l_rp = QuantumRegister(len_width, "l_rp")
    l_s = QuantumRegister(len_width, "l_s")
    Zero = QuantumRegister(2, "Zero")
    qc = QuantumCircuit(Phase1, Phase2, Sign, l_q, l_rp, l_s, Zero, name=name)

    sign_lq = l_q[-1]
    sign_lrp = l_rp[-1]
    sign_ls = l_s[-1]

    condA = Zero[0]
    qc.x(sign_lrp)
    qc.ccx(sign_lq, sign_lrp, condA)
    qc.x(sign_lrp)

    tmp = Zero[1]
    qc.cx(Sign[0], tmp)
    qc.cx(Phase1[0], tmp)
    qc.ccx(condA, tmp, Phase2[0])
    qc.cx(Phase1[0], tmp)
    qc.cx(Sign[0], tmp)

    qc.ccx(condA, Phase2[0], Sign[0])

    qc.x(sign_lrp)
    qc.ccx(sign_lq, sign_lrp, condA)
    qc.x(sign_lrp)

    qc.cx(sign_ls, Phase1[0])
    qc.cx(sign_ls, Phase2[0])

    return qc.to_gate()


def or_latch(qc, latch, cond, tmp):
    qc.x(latch)
    qc.ccx(cond, latch, tmp)
    qc.x(latch)
    qc.cx(tmp, latch)
    qc.x(latch)
    qc.ccx(cond, latch, tmp)
    qc.x(latch)


def _U_len_update_gate(
        *,
        n: int,
        k: int,
        K: int,
        len_width: int,
        work_size: int,
        name: str = "U_LEN",
) -> Gate:
    Work1 = QuantumRegister(work_size, "Work1")
    Work2 = QuantumRegister(work_size, "Work2")
    l_s = QuantumRegister(len_width, "l_s")
    l_q = QuantumRegister(len_width, "l_q")
    l_rp = QuantumRegister(len_width, "l_rp")

    zero_bits = 2 * len_width + 4
    Zero = QuantumRegister(zero_bits, "Zero")

    qc = QuantumCircuit(Work1, Work2, l_s, l_q, l_rp, Zero, name=name)

    latch_u = Zero[0]
    latch_v = Zero[1]
    tmp_cond = Zero[2]
    tmp_or = Zero[3]

    scratch_const = list(Zero[4:4 + (len_width + 2)])
    mcx_pool = list(Zero[4 + (len_width + 2):])

    carry = scratch_const[len_width]
    zflag = scratch_const[len_width + 1]

    sign_ls = l_s[-1]
    sign_lq = l_q[-1]
    sign_lrp = l_rp[-1]

    K_star = (n + 4) - K

    subK = constant_sub(len_width, K_star, len_sub_gate(len_width))
    qc.append(subK, scratch_const + list(l_rp))

    for i in range(K, k - 1, -1):
        u_i = Work1[i - 1]
        v_i = Work2[i - 1]

        mcx_vchain(qc, [u_i, sign_ls, sign_lrp], tmp_cond, mcx_pool)
        or_latch(qc, latch_u, tmp_cond, tmp_or)
        mcx_vchain(qc, [u_i, sign_ls, sign_lrp], tmp_cond, mcx_pool)
        controlled_add_one_1ctrl(qc, latch_u, l_s, mcx_pool)

        mcx_vchain(qc, [v_i, sign_lq, sign_lrp], tmp_cond, mcx_pool)
        or_latch(qc, latch_v, tmp_cond, tmp_or)
        mcx_vchain(qc, [v_i, sign_lq, sign_lrp], tmp_cond, mcx_pool)
        controlled_add_one_1ctrl(qc, latch_v, l_q, mcx_pool)

        controlled_sub_one(qc, l_rp, [sign_lrp], mcx_pool)

    qc.append(len_sub_gate(len_width), list(l_q) + list(l_s) + [carry] + [zflag])

    return qc.to_gate()


def len_update_gate(
        *,
        n: int,
        k: int,
        K: int,
        len_width: int,
        work_size: int,
        target: str,
        name: str,
) -> Gate:
    Ctrl = QuantumRegister(1, "Ctrl")
    Work1 = QuantumRegister(work_size, "Work1")
    Work2 = QuantumRegister(work_size, "Work2")
    l_s = QuantumRegister(len_width, "l_s")
    l_q = QuantumRegister(len_width, "l_q")
    l_rp = QuantumRegister(len_width, "l_rp")
    l_t = QuantumRegister(len_width, "l_t")

    zero_bits = 2 * len_width + 4
    Zero = QuantumRegister(zero_bits, "Zero")
    CarryAdd = QuantumRegister(1, "CarryAdd")

    qc = QuantumCircuit(Ctrl, Work1, Work2, l_s, l_q, l_rp, l_t, Zero, CarryAdd, name=name)

    U = _U_len_update_gate(n=n, k=k, K=K, len_width=len_width, work_size=work_size)
    qc.append(U, list(Work1) + list(Work2) + list(l_s) + list(l_q) + list(l_rp) + list(Zero))

    add_no_z = cuccaro_add_mod_2n_no_z_1ctrl_gate(len_width)

    if target == "l_t":
        tgt = l_t
    elif target == "l_rp":
        tgt = l_rp
    else:
        raise ValueError("target must be 'l_t' or 'l_rp'")

    pool_size = max(1, len_width)
    pool = list(Zero[:pool_size])
    qc.append(add_no_z, [Ctrl[0]] + list(l_s) + list(tgt) + [CarryAdd[0]] + pool)

    qc.append(U.inverse(), list(Work1) + list(Work2) + list(l_s) + list(l_q) + list(l_rp) + list(Zero))
    return qc.to_gate()


def swap_work_and_len_gate(*, n: int, len_width: int, k4: int, K4: int, k5: int, K5: int,
                           name="STEP7_SWAP_LEN") -> Gate:
    work_size = n + 3

    Ctrl = QuantumRegister(1, "Ctrl")
    Work1 = QuantumRegister(work_size, "Work1")
    Work2 = QuantumRegister(work_size, "Work2")
    l_s = QuantumRegister(len_width, "l_s")
    l_q = QuantumRegister(len_width, "l_q")
    l_t = QuantumRegister(len_width, "l_t")
    l_rp = QuantumRegister(len_width, "l_rp")

    zero_bits = 2 * len_width + 4
    Zero_lt = QuantumRegister(zero_bits, "Zero_lt")
    Zero_rp = QuantumRegister(zero_bits, "Zero_rp")
    Carry_lt = QuantumRegister(1, "Carry_lt")
    Carry_rp = QuantumRegister(1, "Carry_rp")

    qc = QuantumCircuit(Ctrl, Work1, Work2, l_s, l_q, l_t, l_rp, Zero_lt, Carry_lt, Zero_rp, Carry_rp, name=name)

    for i in range(work_size):
        cswap_toffoli(qc, Ctrl[0], Work1[i], Work2[i])

    len_lt = len_update_gate(n=n, k=k4, K=K4, len_width=len_width, work_size=work_size, target="l_t", name="LEN_LT")
    qc.append(len_lt,
              [Ctrl[0]] + list(Work1) + list(Work2) + list(l_s) + list(l_q) + list(l_rp) + list(l_t) + list(Zero_lt) + [
                  Carry_lt[0]])

    len_rp = len_update_gate(n=n, k=k5, K=K5, len_width=len_width, work_size=work_size, target="l_rp", name="LEN_RP")
    qc.append(len_rp,
              [Ctrl[0]] + list(Work1) + list(Work2) + list(l_s) + list(l_q) + list(l_rp) + list(l_t) + list(Zero_rp) + [
                  Carry_rp[0]])

    return qc.to_gate()


def _ceil_safe(x: float, eps: float = 1e-12) -> int:
    return math.ceil(x - eps)


def _floor_safe(x: float, eps: float = 1e-12) -> int:
    return math.floor(x + eps)


def Nmax_steps(n: int) -> int:
    return 4 * math.ceil(C_EEA * n)


def active_windows(n: int, T: int):
    c = C_EEA

    denom1 = 4.0 * c - 1.0
    k1 = max(_ceil_safe((T - (n + 2)) / denom1), 1) + 2
    K1 = n + 3

    denom2 = 4.0 * c - 3.0
    k2 = max(_ceil_safe((T - 3.0 * (n + 2)) / denom2), 1) + 1
    K2 = min(_floor_safe(T / 2.0) + 2, n + 2)

    K3 = min(_ceil_safe(T / 4.0) + 1, n + 1)

    denom4 = 4.0 * c - 4.0
    k4 = max(_ceil_safe((T - 4.0 * (n + 2)) / denom4), 1)
    K4 = min(_floor_safe(T / 4.0 + 3.0), n + 3)

    k5 = _ceil_safe(T / (4.0 * c))
    K5 = min(_floor_safe(T / 4.0 + 4.0), n + 3)

    return {
        "r_addsub": (k1, K1),
        "swap": (k2, K2),
        "t_addsub": (1, K3),
        "len_update_lt": (k4, K4),
        "len_update_lr": (k5, K5),
    }


def append_one_step_T(
        qc: QuantumCircuit,
        *,
        T: int,
        n: int,
        len_width: int,
        Phase1, Phase2, Iter,
        Zero, Sign, Work1, Work2,
        l_t, l_q, l_s, l_rp,
        PreS, PostS, PhaseS,
        Step7Zero_lt, Step7Carry_lt, Step7Zero_rp, Step7Carry_rp,
        Pre, Pre2, Pre3,
):
    work_size = n + 3
    w = active_windows(n, T)
    k1, K1 = w["r_addsub"]
    k2, K2 = w["swap"]
    _, K3 = w["t_addsub"]
    k4, K4 = w["len_update_lt"]
    k5, K5 = w["len_update_lr"]

    # pre-shift
    pre = pre_shift_gate(work_size=work_size, len_width=len_width)
    qc.append(pre, [Phase1[0], Phase2[0]] + list(Work2) + list(l_s) + list(PreS))

    # r-side windows
    Work1_win_r = [Work1[i - 1] for i in range(k1, K1 + 1)]
    Work2_win_r = [Work2[i - 1] for i in range(k1, K1 + 1)]

    # rsub
    rsub = location_controlled_sub_gate(n=n, k=k1, K=K1, len_width=len_width)
    qc.append(rsub, [Phase1[0], Phase2[0]] + list(Zero) + list(Pre) + [Sign[0]] + Work1_win_r + Work2_win_r + list(
        l_t) + list(l_q) + list(l_s))

    # radd
    radd = location_controlled_add_gate(n=n, k=k1, K=K1, len_width=len_width)
    qc.append(radd, [Phase1[0], Phase2[0]] + list(Zero) + [Sign[0]] + Work1_win_r + Work2_win_r + list(l_t) + list(
        l_q) + list(l_s))

    # lc-swap window
    Work1_win_swap = [Work1[i - 1] for i in range(k2, K2 + 1)]
    lcs = location_controlled_swap_gate(k=k2, K=K2, len_width=len_width)
    qc.append(lcs, [Phase1[0], Phase2[0]] + list(Zero) + [Sign[0]] + Work1_win_swap + list(l_t) + list(l_q))

    # t-side windows
    Work1_win_t = [Work1[i - 1] for i in range(1, K3 + 1)]
    Work2_win_t = [Work2[i - 1] for i in range(1, K3 + 1)]

    tsub = location_controlled_sub_gate_single(k=1, K=K3, lt_bits=len_width, zero_bits=7 + len_width)
    qc.append(tsub, [Phase1[0], Phase2[0], Sign[0]] + list(Zero) + list(Pre2) + Work1_win_t + Work2_win_t + list(l_t))

    tadd = location_controlled_add_gate_single(k=1, K=K3, lt_bits=len_width, zero_bits=6 + len_width)
    qc.append(tadd, [Phase1[0], Phase2[0], Sign[0]] + list(Zero) + list(Pre3) + Work1_win_t + Work2_win_t + list(l_t))

    # post-shift
    post = post_shift_gate(work_size=work_size, len_width=len_width)
    qc.append(post, [Phase1[0], Phase2[0]] + list(Work2) + list(l_s) + list(PostS))

    # phase update
    pupdate = phase_update_gate(len_width=len_width)
    qc.append(pupdate, [Phase1[0], Phase2[0], Sign[0]] + list(l_q) + list(l_rp) + list(l_s) + list(PhaseS))

    # length update condition
    if T % 4 == 0:
        sign_lq = l_q[-1]
        sign_ls = l_s[-1]
        both = PhaseS[0]
        qc.ccx(sign_lq, sign_ls, both)

        len_up = swap_work_and_len_gate(n=n, len_width=len_width, k4=k4, K4=K4, k5=k5, K5=K5)
        qc.append(
            len_up,
            [both]
            + list(Work1) + list(Work2)
            + list(l_s) + list(l_q) + list(l_t) + list(l_rp)
            + list(Step7Zero_lt) + [Step7Carry_lt[0]]
            + list(Step7Zero_rp) + [Step7Carry_rp[0]]
        )

        qc.ccx(sign_lq, sign_ls, both)

        # Iter ^= 1 under same condition
        qc.ccx(sign_lq, sign_ls, both)
        qc.cx(both, Iter[0])
        qc.ccx(sign_lq, sign_ls, both)


def make_global_registers(*, n: int, len_width: int):
    work_size = n + 3

    Phase1 = QuantumRegister(1, "Phase1")
    Phase2 = QuantumRegister(1, "Phase2")
    Iter = QuantumRegister(1, "Iter")

    Zero = QuantumRegister(2 + len_width, "Zero")
    Sign = QuantumRegister(1, "Sign")
    Work1 = QuantumRegister(work_size, "Work1")
    Work2 = QuantumRegister(work_size, "Work2")

    l_t = QuantumRegister(len_width, "l_t")
    l_q = QuantumRegister(len_width, "l_q")
    l_s = QuantumRegister(len_width, "l_s")
    l_rp = QuantumRegister(len_width, "l_rp")

    # fixed scratch pools
    PreS = QuantumRegister(len_width + 4, "PreS")
    PostS = QuantumRegister(len_width + 4, "PostS")
    PhaseS = QuantumRegister(2, "PhaseS")

    Step7Zero_lt = QuantumRegister(2 * len_width + 4, "Step7Zero_lt")
    Step7Carry_lt = QuantumRegister(1, "Step7Carry_lt")
    Step7Zero_rp = QuantumRegister(2 * len_width + 4, "Step7Zero_rp")
    Step7Carry_rp = QuantumRegister(1, "Step7Carry_rp")

    Pre = QuantumRegister(3, "Pre")
    Pre2 = QuantumRegister(5, "Pre2")
    Pre3 = QuantumRegister(4, "Pre3")

    return (Phase1, Phase2, Iter,
            Zero, Sign, Work1, Work2,
            l_t, l_q, l_s, l_rp,
            PreS, PostS, PhaseS, Step7Zero_lt, Step7Carry_lt, Step7Zero_rp, Step7Carry_rp, Pre, Pre2, Pre3)


def _freeze_params(params):
    out = []
    for p in params:
        if isinstance(p, (int, float, str, bool, type(None))):
            out.append(p)
        else:
            out.append(repr(p))
    return tuple(out)


def _inst_cache_key(inst: Instruction):
    return (
        inst.name,
        inst.num_qubits,
        inst.num_clbits,
        _freeze_params(getattr(inst, "params", [])),
    )


def count_instruction_ops(inst: Instruction) -> Counter:
    name = inst.name

    if name in PRIMITIVE_OPS:
        return Counter({name: 1})

    key = _inst_cache_key(inst)
    if key in _INST_COUNT_CACHE:
        return _INST_COUNT_CACHE[key].copy()

    if inst.definition is None:
        raise ValueError(
            f"Instruction {name!r} has no definition; cannot recurse-count it."
        )

    total = Counter()
    for subinst, qargs, cargs in inst.definition.data:
        total += count_instruction_ops(subinst)

    _INST_COUNT_CACHE[key] = total.copy()
    return total


def count_circuit_ops_recursive(qc: QuantumCircuit) -> Counter:
    total = Counter()
    for inst, qargs, cargs in qc.data:
        total += count_instruction_ops(inst)
    return total


def _as_result_dict(T: int, module_name: str, ops: Counter):
    return {
        "T": T,
        "module": module_name,
        "ccx": int(ops.get("ccx", 0)),
        "cx": int(ops.get("cx", 0)),
        "x": int(ops.get("x", 0)),
        "total": int(sum(ops.values())),
    }


@lru_cache(maxsize=None)
def count_pre_shift_cached(work_size: int, len_width: int) -> Counter:
    g = pre_shift_gate(work_size=work_size, len_width=len_width)
    return count_instruction_ops(g)


@lru_cache(maxsize=None)
def count_rsub_cached(n: int, k1: int, K1: int, len_width: int) -> Counter:
    g = location_controlled_sub_gate(n=n, k=k1, K=K1, len_width=len_width)
    return count_instruction_ops(g)


@lru_cache(maxsize=None)
def count_radd_cached(n: int, k1: int, K1: int, len_width: int) -> Counter:
    g = location_controlled_add_gate(n=n, k=k1, K=K1, len_width=len_width)
    return count_instruction_ops(g)


@lru_cache(maxsize=None)
def count_lcswap_cached(k2: int, K2: int, len_width: int) -> Counter:
    g = location_controlled_swap_gate(k=k2, K=K2, len_width=len_width)
    return count_instruction_ops(g)


@lru_cache(maxsize=None)
def count_tsub_cached(K3: int, len_width: int) -> Counter:
    g = location_controlled_sub_gate_single(
        k=1, K=K3, lt_bits=len_width, zero_bits=7 + len_width
    )
    return count_instruction_ops(g)


@lru_cache(maxsize=None)
def count_tadd_cached(K3: int, len_width: int) -> Counter:
    g = location_controlled_add_gate_single(
        k=1, K=K3, lt_bits=len_width, zero_bits=6 + len_width
    )
    return count_instruction_ops(g)


@lru_cache(maxsize=None)
def count_post_shift_cached(work_size: int, len_width: int) -> Counter:
    g = post_shift_gate(work_size=work_size, len_width=len_width)
    return count_instruction_ops(g)


@lru_cache(maxsize=None)
def count_phase_update_cached(len_width: int) -> Counter:
    g = phase_update_gate(len_width=len_width)
    return count_instruction_ops(g)


@lru_cache(maxsize=None)
def count_len_update_inner_cached(
        n: int, len_width: int, k4: int, K4: int, k5: int, K5: int
) -> Counter:
    g = swap_work_and_len_gate(
        n=n, len_width=len_width, k4=k4, K4=K4, k5=k5, K5=K5
    )
    return count_instruction_ops(g)


def count_len_update_outer_wrapper_ops() -> Counter:
    return Counter({"ccx": 4, "cx": 1})


def count_one_step_by_modules(*, n: int, len_width: int, T: int):
    work_size = n + 3
    w = active_windows(n, T)
    k1, K1 = w["r_addsub"]
    k2, K2 = w["swap"]
    _, K3 = w["t_addsub"]
    k4, K4 = w["len_update_lt"]
    k5, K5 = w["len_update_lr"]

    module_rows = []
    step_total = Counter()

    def add_module(module_name: str, ops: Counter):
        nonlocal step_total, module_rows
        step_total += ops
        module_rows.append(_as_result_dict(T, module_name, ops))

    add_module("pre_shift", count_pre_shift_cached(work_size, len_width))
    add_module("rsub", count_rsub_cached(n, k1, K1, len_width))
    add_module("radd", count_radd_cached(n, k1, K1, len_width))
    add_module("lc_swap", count_lcswap_cached(k2, K2, len_width))
    add_module("tsub", count_tsub_cached(K3, len_width))
    add_module("tadd", count_tadd_cached(K3, len_width))
    add_module("post_shift", count_post_shift_cached(work_size, len_width))
    add_module("phase_update", count_phase_update_cached(len_width))

    if T % 4 == 0:
        add_module("len_update_outer", count_len_update_outer_wrapper_ops())
        add_module(
            "len_update_inner",
            count_len_update_inner_cached(n, len_width, k4, K4, k5, K5),
        )

    return step_total, module_rows


def count_all_steps_by_modules(
        *,
        n: int,
        len_width: int,
        T_max: int,
        log_every: int = 50,
        save_step_file: str | None = None,
        save_module_file: str | None = None,
):
    grand_total = Counter()
    t0 = time.perf_counter()

    step_f = open(save_step_file, "w", encoding="utf-8") if save_step_file else None
    mod_f = open(save_module_file, "w", encoding="utf-8") if save_module_file else None

    try:
        for T in range(1, T_max + 1):
            step_total, module_rows = count_one_step_by_modules(
                n=n, len_width=len_width, T=T
            )
            grand_total += step_total

            step_row = {
                "T": T,
                "ccx": int(step_total.get("ccx", 0)),
                "cx": int(step_total.get("cx", 0)),
                "x": int(step_total.get("x", 0)),
                "total": int(sum(step_total.values())),
            }

            if step_f:
                step_f.write(f"{step_row}\n")

            if mod_f:
                for row in module_rows:
                    mod_f.write(f"{row}\n")

            if T % log_every == 0 or T == T_max:
                print(
                    f"[{T}/{T_max}] "
                    f"step_ccx={step_row['ccx']} "
                    f"step_cx={step_row['cx']} "
                    f"step_x={step_row['x']} "
                    f"step_total={step_row['total']} | "
                    f"grand_ccx={grand_total['ccx']} "
                    f"grand_cx={grand_total['cx']} "
                    f"grand_x={grand_total['x']} "
                    f"grand_total={grand_total['total']}",
                    flush=True,
                )

            del step_total, module_rows, step_row
            if T % 20 == 0:
                gc.collect()

    finally:
        if step_f:
            step_f.close()
        if mod_f:
            mod_f.close()

    t1 = time.perf_counter()

    print("\n=== FINAL ===")
    print("Toffoli count (ccx) =", int(grand_total.get("ccx", 0)))
    print("CX count            =", int(grand_total.get("cx", 0)))
    print("X count             =", int(grand_total.get("x", 0)))
    print("Total ops           =", int(sum(grand_total.values())))
    print("Elapsed (s)         =", t1 - t0)

    return {
        "ccx": int(grand_total.get("ccx", 0)),
        "cx": int(grand_total.get("cx", 0)),
        "x": int(grand_total.get("x", 0)),
        "total": int(sum(grand_total.values())),
        "elapsed_s": t1 - t0,
    }


def report_first_steps_modules(*, n: int, len_width: int, steps: int = 10):
    for T in range(1, steps + 1):
        step_total, rows = count_one_step_by_modules(n=n, len_width=len_width, T=T)
        print(f"\n=== T={T} ===")
        for row in rows:
            print(
                f"{row['module']:18s} "
                f"ccx={row['ccx']:8d} "
                f"cx={row['cx']:8d} "
                f"x={row['x']:8d} "
                f"total={row['total']:8d}"
            )
        print(
            f"{'STEP_TOTAL':18s} "
            f"ccx={int(step_total.get('ccx', 0)):8d} "
            f"cx={int(step_total.get('cx', 0)):8d} "
            f"x={int(step_total.get('x', 0)):8d} "
            f"total={int(sum(step_total.values())):8d}"
        )


def get_n_config(n: int) -> dict:
    if n not in N_CONFIG:
        raise ValueError(
            f"Unsupported n={n}. Supported n are: {sorted(N_CONFIG.keys())}"
        )
    return N_CONFIG[n].copy()


def choose_mode(n: int) -> Literal["full", "recursive"]:
    if n <= 256:
        return "full"
    if n in (384, 512):
        return "recursive"
    raise ValueError(f"Unsupported n={n} for mode selection.")


def build_full_steps_circuit(n: int, len_width: int, T_max: int) -> QuantumCircuit:
    regs = make_global_registers(n=n, len_width=len_width)
    qc_full = QuantumCircuit(*regs, name="FULL_STEPS")

    (Phase1, Phase2, Iter,
     Zero, Sign, Work1, Work2,
     l_t, l_q, l_s, l_rp,
     PreS, PostS, PhaseS,
     Step7Zero_lt, Step7Carry_lt, Step7Zero_rp, Step7Carry_rp,
     Pre, Pre2, Pre3) = regs

    for T in range(T_max):
        append_one_step_T(
            qc_full,
            T=T + 1, n=n, len_width=len_width,
            Phase1=Phase1, Phase2=Phase2, Iter=Iter,
            Zero=Zero, Sign=Sign, Work1=Work1, Work2=Work2,
            l_t=l_t, l_q=l_q, l_s=l_s, l_rp=l_rp,
            PreS=PreS, PostS=PostS, PhaseS=PhaseS,
            Step7Zero_lt=Step7Zero_lt, Step7Carry_lt=Step7Carry_lt,
            Step7Zero_rp=Step7Zero_rp, Step7Carry_rp=Step7Carry_rp,
            Pre=Pre, Pre2=Pre2, Pre3=Pre3,
        )

    return qc_full


def count_full_circuit_ops(
        *,
        n: int,
        len_width: int,
        T_max: int,
        save_qasm: bool = False,
        qasm_path: Optional[str] = None,
) -> dict:
    qc_full = build_full_steps_circuit(n=n, len_width=len_width, T_max=T_max)

    if save_qasm:
        if qasm_path is None:
            qasm_path = f"n{n}_full_steps.qasm"
        with open(qasm_path, "w", encoding="utf-8") as f:
            f.write(qasm2_dumps(qc_full))

    t0 = time.perf_counter()

    qc_toff = transpile(
        qc_full,
        basis_gates=["ccx", "cx", "x"],
        optimization_level=0,
        layout_method=None,
        routing_method=None,
        scheduling_method=None,
    )
    ops = qc_toff.count_ops()

    t1 = time.perf_counter()

    result = {
        "mode": "full",
        "n": n,
        "len_width": len_width,
        "T_max": T_max,
        "ccx": int(ops.get("ccx", 0)),
        "cx": int(ops.get("cx", 0)),
        "x": int(ops.get("x", 0)),
        "total": int(sum(ops.values())),
        "elapsed_s": t1 - t0,
    }

    print("\n=== FINAL (FULL) ===")
    print("n                 =", result["n"])
    print("len_width         =", result["len_width"])
    print("T_max             =", result["T_max"])
    print("Toffoli count     =", result["ccx"])
    print("CX count          =", result["cx"])
    print("X count           =", result["x"])
    print("Total ops         =", result["total"])
    print("Elapsed (s)       =", result["elapsed_s"])

    return result


def run_for_n(
        n: int,
        *,
        mode: Optional[Literal["auto", "full", "recursive"]] = "auto",
        inspect_first_steps: int = 0,
        log_every: int = 20,
        output_dir: str = ".",
        save_qasm_for_full: bool = False,
) -> dict:
    cfg = get_n_config(n)
    len_width = cfg["len_width"]
    T_max = cfg["T_max"]

    if mode == "auto":
        resolved_mode = choose_mode(n)
    else:
        resolved_mode = mode

    if resolved_mode == "full" and n > 256:
        raise ValueError(f"Requested mode='full' for n={n}, but by policy n>256 should use recursive mode.")
    if resolved_mode == "recursive" and n <= 256:
        print(f"[WARN] n={n} normally uses full mode, but recursive mode was explicitly requested.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[RUN] n={n}, len_width={len_width}, T_max={T_max}, mode={resolved_mode}")

    if resolved_mode == "full":
        qasm_path = str(output_dir / f"n{n}_full_steps.qasm") if save_qasm_for_full else None
        return count_full_circuit_ops(
            n=n,
            len_width=len_width,
            T_max=T_max,
            save_qasm=save_qasm_for_full,
            qasm_path=qasm_path,
        )

    if resolved_mode == "recursive":
        if inspect_first_steps > 0:
            report_first_steps_modules(
                n=n,
                len_width=len_width,
                steps=inspect_first_steps,
            )

        return count_all_steps_by_modules(
            n=n,
            len_width=len_width,
            T_max=T_max,
            log_every=log_every,
            save_step_file=str(output_dir / f"n{n}_step_counts_recursive.txt"),
            save_module_file=str(output_dir / f"n{n}_module_counts_recursive.txt"),
        )

    raise ValueError(f"Unknown mode={mode}")


def main():
    parser = argparse.ArgumentParser(description="Run modular inversion resource estimation")

    parser.add_argument("--n", type=int, required=True,
                        help="problem size n (e.g., 64,128,...,512)")

    parser.add_argument("--mode", type=str, default="auto",
                        choices=["auto", "full", "recursive"],
                        help="execution mode")

    parser.add_argument("--inspect", type=int, default=0,
                        help="print first K steps (only for recursive)")

    parser.add_argument("--log_every", type=int, default=20,
                        help="logging frequency (recursive mode)")

    parser.add_argument("--outdir", type=str, default="outputs",
                        help="output directory")

    parser.add_argument("--save_qasm", action="store_true",
                        help="save full circuit qasm (only for full mode)")

    args = parser.parse_args()

    result = run_for_n(
        args.n,
        mode=args.mode,
        inspect_first_steps=args.inspect,
        log_every=args.log_every,
        output_dir=args.outdir,
        save_qasm_for_full=args.save_qasm,
    )

    print("\nFINAL RESULT:")
    print(result)


if __name__ == "__main__":
    main()
