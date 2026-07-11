from functools import lru_cache
from typing import Literal, Optional, Sequence

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate, Qubit

import eea_circuit_updated as _e

C_EEA = _e.C_EEA
N_CONFIG = _e.N_CONFIG
paper_len_width = _e.paper_len_width
paper_shift_width = _e.paper_shift_width
Nmax_steps = _e.Nmax_steps
active_windows = _e.active_windows
get_n_config = getattr(_e, "get_n_config")
set_measurement_uncompute = _e.set_measurement_uncompute
count_circuit_ops_recursive = getattr(_e, "count_circuit_ops_recursive", None)


def __getattr__(name: str):
    return getattr(_e, name)


def _tight_unary_depth_for_labels(labels: Sequence[int]) -> int:
    labels = sorted(set(labels))
    if len(labels) <= 1:
        return 0
    bit = _e._split_bit(labels)
    z = [x for x in labels if ((x >> bit) & 1) == 0]
    o = [x for x in labels if ((x >> bit) & 1) == 1]
    return 1 + max(_tight_unary_depth_for_labels(z), _tight_unary_depth_for_labels(o))


def unary_iteration_tight(qc: QuantumCircuit, *, index_reg: Sequence[Qubit], labels: Sequence[int],
                          ctrl: Qubit, ancillas: Sequence[Qubit], leaf_fn, order: Literal["inc", "dec"] = "inc") -> None:
    labels = sorted(set(labels))
    if not labels:
        return
    need = _tight_unary_depth_for_labels(labels)
    if len(ancillas) < need:
        raise ValueError(f"tight unary iteration needs {need} ancillas, got {len(ancillas)}")
    def rec(sub_labels, g, depth):
        if len(sub_labels) == 1:
            leaf_fn(sub_labels[0], g); return
        b = _e._split_bit(sub_labels)
        z = [x for x in sub_labels if ((x >> b) & 1) == 0]
        o = [x for x in sub_labels if ((x >> b) & 1) == 1]
        h = ancillas[depth]
        _e._and_with_index_bit(qc, g, index_reg[b], h, 0)
        if order == "inc":
            rec(z, h, depth+1)
            qc.cx(g, h)
            rec(o, h, depth+1)
            qc.cx(g, h)
        else:
            qc.cx(g, h)
            rec(o, h, depth+1)
            qc.cx(g, h)
            rec(z, h, depth+1)
        _e._uncompute_and_with_index_bit(qc, g, index_reg[b], h, 0)
    rec(labels, ctrl, 0)


def dual_unary_iteration_tight(qc: QuantumCircuit, *, index_a: Sequence[Qubit], index_b: Sequence[Qubit], labels: Sequence[int],
                               ctrl_a: Qubit, ctrl_b: Qubit, ancillas_a: Sequence[Qubit], ancillas_b: Sequence[Qubit],
                               leaf_fn, order: Literal["inc", "dec"] = "inc") -> None:
    labels = sorted(set(labels))
    if not labels:
        return
    need = _tight_unary_depth_for_labels(labels)
    if len(ancillas_a) < need or len(ancillas_b) < need:
        raise ValueError(f"tight dual unary iteration needs {need} ancillas per endpoint")
    def rec(sub_labels, ga, gb, depth):
        if len(sub_labels) == 1:
            leaf_fn(sub_labels[0], ga, gb); return
        bit = _e._split_bit(sub_labels)
        z = [x for x in sub_labels if ((x >> bit) & 1) == 0]
        o = [x for x in sub_labels if ((x >> bit) & 1) == 1]
        ha = ancillas_a[depth]; hb = ancillas_b[depth]
        _e._and_with_index_bit(qc, ga, index_a[bit], ha, 0)
        _e._and_with_index_bit(qc, gb, index_b[bit], hb, 0)
        if order == "inc":
            rec(z, ha, hb, depth+1)
            qc.cx(ga, ha); qc.cx(gb, hb)
            rec(o, ha, hb, depth+1)
            qc.cx(gb, hb); qc.cx(ga, ha)
        else:
            qc.cx(ga, ha); qc.cx(gb, hb)
            rec(o, ha, hb, depth+1)
            qc.cx(gb, hb); qc.cx(ga, ha)
            rec(z, ha, hb, depth+1)
        _e._uncompute_and_with_index_bit(qc, gb, index_b[bit], hb, 0)
        _e._uncompute_and_with_index_bit(qc, ga, index_a[bit], ha, 0)
    rec(labels, ctrl_a, ctrl_b, 0)


def _toggle_eq_const_under_ctrl_direct(qc: QuantumCircuit, *, endpoint: Sequence[Qubit], const: int, ctrl: Qubit, acc: Qubit, scratch: Sequence[Qubit]) -> None:
    # scratch supplies a temporary eq flag followed by mcx scratch.
    eq = scratch[0]
    pool = list(scratch[1:])
    _e.compute_eq_const(qc, endpoint, const, eq, pool)
    qc.ccx(ctrl, eq, acc)
    _e.compute_eq_const(qc, endpoint, const, eq, pool)


def _const_scratch(Scratch, width: int, carry: Qubit) -> list[Qubit]:
    # add_const_mod_2n expects width constant bits followed by one clean carry.
    return list(Scratch[:width]) + [carry]


def _dirty_c3x(qc: QuantumCircuit, a: Qubit, b: Qubit, c: Qubit, target: Qubit, dirty: Qubit) -> None:
    """Exact C^3X using one dirty ancilla and four Toffolis.

    The dirty qubit is restored even when it initially contains an unknown value.
    """
    qc.ccx(a, b, dirty)
    qc.ccx(c, dirty, target)
    qc.ccx(a, b, dirty)
    qc.ccx(c, dirty, target)


def _controlled_toffoli_dirty(qc: QuantumCircuit, ctrl: Qubit, a: Qubit, b: Qubit, target: Qubit, dirty: Qubit) -> None:
    _dirty_c3x(qc, ctrl, a, b, target, dirty)


def controlled_maj_dirty(qc: QuantumCircuit, ctrl: Qubit, a: Qubit, b: Qubit, c: Qubit, dirty: Qubit) -> None:
    qc.cx(a, b)
    qc.cx(a, c)
    _controlled_toffoli_dirty(qc, ctrl, c, b, a, dirty)


def controlled_uma_dirty(qc: QuantumCircuit, ctrl: Qubit, a: Qubit, b: Qubit, c: Qubit, dirty: Qubit) -> None:
    _controlled_toffoli_dirty(qc, ctrl, c, b, a, dirty)
    qc.cx(a, c)
    qc.cx(c, b)


def controlled_maj_inv_dirty(qc: QuantumCircuit, ctrl: Qubit, a: Qubit, b: Qubit, c: Qubit, dirty: Qubit) -> None:
    _controlled_toffoli_dirty(qc, ctrl, c, b, a, dirty)
    qc.cx(a, c)
    qc.cx(a, b)


def controlled_uma_inv_dirty(qc: QuantumCircuit, ctrl: Qubit, a: Qubit, b: Qubit, c: Qubit, dirty: Qubit) -> None:
    qc.cx(c, b)
    qc.cx(a, c)
    _controlled_toffoli_dirty(qc, ctrl, c, b, a, dirty)


def _apply_cell_dirty(qc: QuantumCircuit, mode: Literal["add", "sub"], pass_kind: Literal["first", "second"],
                      ctrl: Qubit, addend: Qubit, target: Qubit, carry: Qubit, dirty: Qubit) -> None:
    if mode == "add" and pass_kind == "first":
        controlled_maj_dirty(qc, ctrl, addend, target, carry, dirty)
    elif mode == "add" and pass_kind == "second":
        controlled_uma_dirty(qc, ctrl, addend, target, carry, dirty)
    elif mode == "sub" and pass_kind == "first":
        controlled_uma_inv_dirty(qc, ctrl, addend, target, carry, dirty)
    elif mode == "sub" and pass_kind == "second":
        controlled_maj_inv_dirty(qc, ctrl, addend, target, carry, dirty)
    else:
        raise ValueError("bad arithmetic cell mode/pass")


@lru_cache(maxsize=None)
def lc_swap_unary_gate(*, k: int, K: int, len_width: int, name: str = "LC_SWAP_S835_FAST") -> Gate:
    if k > K:
        raise ValueError("need k <= K")
    M = K - k + 1
    depth = _e.unary_depth(M)
    base = max(len_width, depth)
    scratch_size = base + 1
    Ctrl = QuantumRegister(1, "Ctrl")
    Sign = QuantumRegister(1, "Sign")
    Work1 = QuantumRegister(M, "Work1")
    l_t = QuantumRegister(len_width, "l_t")
    l_q = QuantumRegister(len_width, "l_q")
    Scratch = QuantumRegister(scratch_size, "Scratch")
    qc = _e._block_circuit(Ctrl, Sign, Work1, l_t, l_q, Scratch, name=name)
    carry = Scratch[base]
    cs = list(Scratch[:len_width]) + [carry]
    qc.append(_e.cuccaro_add_mod_2n_no_z_gate(len_width, name="ADD_lt_to_lq"), list(l_t) + list(l_q) + [carry])
    _e.add_const_mod_2n(qc, l_q, 3, cs)
    path = list(Scratch[:depth])
    def leaf(j: int, ej: Qubit) -> None:
        _e.cswap_toffoli(qc, ej, Sign[0], Work1[j - k])
    unary_iteration_tight(qc, index_reg=l_q, labels=list(range(k, K + 1)), ctrl=Ctrl[0], ancillas=path, leaf_fn=leaf, order="inc")
    _e.sub_const_mod_2n(qc, l_q, 3, cs)
    qc.append(_e.cuccaro_sub_mod_2n_no_z_gate(len_width, name="SUB_lt_from_lq"), list(l_t) + list(l_q) + [carry])
    return _e._finalize_block(qc)


@lru_cache(maxsize=None)
def lc_interval_addsub_unary_gate(*, n: int, k: int, K: int, len_width: int, shift_width: int,
                                  mode: Literal["add", "sub"], sign_update: bool,
                                  target: Literal["work1", "work2"], name: str) -> Gate:
    if k > K:
        raise ValueError("need k <= K")
    M = K - k + 1
    endpoint_width = max(len_width, shift_width)
    # If the interval has one more label than a power of two (the n=256 worst case),
    # handle the top label separately and run the unary scans over the remaining power-of-two interval.
    labels_all_abs = list(range(k, K + 1))
    rel_count = len(labels_all_abs)
    labels_main = list(range(rel_count))
    top_special = False
    if rel_count > 1 and ((rel_count - 1) & (rel_count - 2)) == 0:
        labels_main = list(range(rel_count - 1))
        top_special = True
    top_rel = rel_count - 1
    depth = _tight_unary_depth_for_labels(labels_main)
    # Layout note:
    #   anc_a/anc_b occupy the first 2*depth wires and are used only by
    #   the unary endpoint scans.  Endpoint affine transforms need
    #   endpoint_width scratch wires plus a carry.  For late steps the unary
    #   depth can be smaller than endpoint_width; placing carry immediately
    #   after the unary paths would then alias it with the constant-adder
    #   scratch.  We therefore place carry/acc/cell_pool after the larger of
    #   the unary-scratch region and the endpoint-transform scratch region.
    base = max(2 * depth, endpoint_width)
    scratch_size = base + 3
    Ctrl = QuantumRegister(1, "Ctrl")
    Sign = QuantumRegister(1, "Sign")
    Work1 = QuantumRegister(M, "Work1")
    Work2 = QuantumRegister(M, "Work2")
    l_t = QuantumRegister(len_width, "l_t")
    l_q = QuantumRegister(len_width, "l_q")
    l_s = QuantumRegister(shift_width, "l_s")
    Scratch = QuantumRegister(scratch_size, "Scratch")
    qc = _e._block_circuit(Ctrl, Sign, Work1, Work2, l_t, l_q, l_s, Scratch, name=name)
    anc_a = list(Scratch[:depth])
    anc_b = list(Scratch[depth:2*depth])
    carry = Scratch[base]
    acc = Scratch[base + 1]
    cell_pool = [Scratch[base + 2]]
    # Top-special equality controls need a clean v-chain scratch pool.  At the
    # moment they are used, all wires before 'base' are clean.
    eq_scratch = [Scratch[base + 2]] + list(Scratch[:base])
    cs = _const_scratch(Scratch, endpoint_width, carry)
    # Prepare L=(ell_t-1)+(ell_q-1)+4 and R=n+2-(ell_s-1).
    qc.append(_e.cuccaro_add_mod_2n_no_z_gate(len_width, name="ADD_lt_to_lq"), list(l_t) + list(l_q) + [carry])
    _e.add_const_mod_2n(qc, l_q, 4, cs[:len_width] + [carry])
    _e.const_minus_inplace(qc, l_s, n + 2, cs[:shift_width] + [carry])
    # Convert absolute endpoints to relative offsets in [0, K-k].
    _e.sub_const_mod_2n(qc, l_q, k, cs[:len_width] + [carry])
    _e.sub_const_mod_2n(qc, l_s, k, cs[:shift_width] + [carry])
    def qpair(j: int) -> tuple[Qubit, Qubit]:
        j_abs = k + j
        idx = j_abs - k
        if target == "work1":
            return Work2[idx], Work1[idx]
        if target == "work2":
            return Work1[idx], Work2[idx]
        raise ValueError("bad target")
    def leaf_first(j: int, rj: Qubit, lj: Qubit) -> None:
        addend, tgt = qpair(j)
        qc.cx(rj, acc)
        _e._apply_cell(qc, mode, "first", acc, addend, tgt, carry, cell_pool)
        qc.cx(lj, acc)
    if top_special:
        _toggle_eq_const_under_ctrl_direct(qc, endpoint=l_s, const=top_rel, ctrl=Ctrl[0], acc=acc, scratch=eq_scratch)
        addend, tgt = qpair(top_rel)
        _e._apply_cell(qc, mode, "first", acc, addend, tgt, carry, cell_pool)
        _toggle_eq_const_under_ctrl_direct(qc, endpoint=l_q, const=top_rel, ctrl=Ctrl[0], acc=acc, scratch=eq_scratch)
    dual_unary_iteration_tight(qc, index_a=l_s, index_b=l_q, labels=labels_main,
                            ctrl_a=Ctrl[0], ctrl_b=Ctrl[0], ancillas_a=anc_a,
                            ancillas_b=anc_b, leaf_fn=leaf_first, order="dec")
    if sign_update:
        qc.cx(carry, Sign[0])
    def leaf_second(j: int, rj: Qubit, lj: Qubit) -> None:
        addend, tgt = qpair(j)
        qc.cx(lj, acc)
        _e._apply_cell(qc, mode, "second", acc, addend, tgt, carry, cell_pool)
        qc.cx(rj, acc)
    dual_unary_iteration_tight(qc, index_a=l_s, index_b=l_q, labels=labels_main,
                            ctrl_a=Ctrl[0], ctrl_b=Ctrl[0], ancillas_a=anc_a,
                            ancillas_b=anc_b, leaf_fn=leaf_second, order="inc")
    if top_special:
        _toggle_eq_const_under_ctrl_direct(qc, endpoint=l_q, const=top_rel, ctrl=Ctrl[0], acc=acc, scratch=eq_scratch)
        addend, tgt = qpair(top_rel)
        _e._apply_cell(qc, mode, "second", acc, addend, tgt, carry, cell_pool)
        _toggle_eq_const_under_ctrl_direct(qc, endpoint=l_s, const=top_rel, ctrl=Ctrl[0], acc=acc, scratch=eq_scratch)
    _e.add_const_mod_2n(qc, l_s, k, cs[:shift_width] + [carry])
    _e.add_const_mod_2n(qc, l_q, k, cs[:len_width] + [carry])
    _e.const_minus_inplace(qc, l_s, n + 2, cs[:shift_width] + [carry])
    _e.sub_const_mod_2n(qc, l_q, 4, cs[:len_width] + [carry])
    qc.append(_e.cuccaro_sub_mod_2n_no_z_gate(len_width, name="SUB_lt_from_lq"), list(l_t) + list(l_q) + [carry])
    return _e._finalize_block(qc)


@lru_cache(maxsize=None)
def lc_prefix_addsub_unary_gate(*, k: int, K: int, len_width: int,
                                mode: Literal["add", "sub"], sign_update: bool,
                                target: Literal["work1", "work2"], name: str) -> Gate:
    if k > K:
        raise ValueError("need k <= K")
    M = K - k + 1
    depth = _e.unary_depth(M)
    base = max(depth, len_width)
    scratch_size = base + 3
    Ctrl = QuantumRegister(1, "Ctrl")
    Sign = QuantumRegister(1, "Sign")
    Work1 = QuantumRegister(M, "Work1")
    Work2 = QuantumRegister(M, "Work2")
    l_t = QuantumRegister(len_width, "l_t")
    Scratch = QuantumRegister(scratch_size, "Scratch")
    qc = _e._block_circuit(Ctrl, Sign, Work1, Work2, l_t, Scratch, name=name)
    path = list(Scratch[:depth])
    carry = Scratch[base]
    acc = Scratch[base + 1]
    cell_pool = [Scratch[base + 2]]
    cs = list(Scratch[:len_width]) + [carry]
    _e.add_const_mod_2n(qc, l_t, 2, cs)
    def qpair(j: int) -> tuple[Qubit, Qubit]:
        idx = j - k
        if target == "work1":
            return Work2[idx], Work1[idx]
        if target == "work2":
            return Work1[idx], Work2[idx]
        raise ValueError("bad target")
    def leaf_first(j: int, ej: Qubit) -> None:
        addend, tgt = qpair(j)
        qc.cx(ej, acc)
        _e._apply_cell(qc, mode, "first", acc, addend, tgt, carry, cell_pool)
        if j == k:
            qc.cx(Ctrl[0], acc)
    unary_iteration_tight(qc, index_reg=l_t, labels=list(range(k, K + 1)), ctrl=Ctrl[0], ancillas=path, leaf_fn=leaf_first, order="dec")
    if sign_update:
        qc.cx(carry, Sign[0])
    qc.cx(Ctrl[0], acc)
    def leaf_second(j: int, ej: Qubit) -> None:
        addend, tgt = qpair(j)
        _e._apply_cell(qc, mode, "second", acc, addend, tgt, carry, cell_pool)
        qc.cx(ej, acc)
    unary_iteration_tight(qc, index_reg=l_t, labels=list(range(k, K + 1)), ctrl=Ctrl[0], ancillas=path, leaf_fn=leaf_second, order="inc")
    _e.sub_const_mod_2n(qc, l_t, 2, cs)
    return _e._finalize_block(qc)

# Reuse the low-aux length update; it is already the paper dirty-work construction with live-range shared scratch.
import eea_circuit_s835_lowaux as _low
len_update_lt_unary_gate = _low.len_update_lt_unary_gate
len_update_lrp_unary_gate = _low.len_update_lrp_unary_gate

@lru_cache(maxsize=None)
def swap_work_and_len_unary_shared_gate(*, n: int, len_width: int, k4: int, K4: int,
                                        k5: int, K5: int, name: str = "SWAP_AND_LEN_S835_FAST") -> Gate:
    work_size = n + 3
    depth4 = _e.unary_depth(K4 - k4 + 1)
    depth5 = _e.unary_depth(K5 - k5 + 1)
    scratch4 = max(len_width + 1, depth4 + 2)
    scratch5 = max(len_width + 1, depth5 + 2)
    scratch_size = max(scratch4, scratch5)
    Ctrl = QuantumRegister(1, "Ctrl")
    Work1 = QuantumRegister(work_size, "Work1")
    Work2 = QuantumRegister(work_size, "Work2")
    l_t = QuantumRegister(len_width, "l_t")
    l_rp = QuantumRegister(len_width, "l_rp")
    Scratch = QuantumRegister(scratch_size, "Scratch")
    qc = _e._block_circuit(Ctrl, Work1, Work2, l_t, l_rp, Scratch, name=name)
    for i in range(work_size):
        _e.cswap_toffoli(qc, Ctrl[0], Work1[i], Work2[i])
    gate_lt = len_update_lt_unary_gate(n=n, k=k4, K=K4, len_width=len_width)
    _e._append_with_optional_clbits(qc, gate_lt, [Ctrl[0]] + list(Work1[k4 - 1:K4]) + list(Work2[k4 - 1:K4])
                                    + list(l_t) + list(l_rp) + list(Scratch[:scratch4]))
    gate_lrp = len_update_lrp_unary_gate(n=n, k=k5, K=K5, len_width=len_width)
    _e._append_with_optional_clbits(qc, gate_lrp, [Ctrl[0]] + list(Work1[k5 - 1:K5]) + list(Work2[k5 - 1:K5])
                                    + list(l_t) + list(l_rp) + list(Scratch[:scratch5]))
    return _e._finalize_block(qc)


def _fastdual_interval_scratch_size(n: int, k: int, K: int, len_width: int, shift_width: int) -> int:
    """Scratch size used by ``lc_interval_addsub_unary_gate``.

    This helper mirrors the scratch layout in ``lc_interval_addsub_unary_gate``.
    It is intentionally kept next to ``qiskit_paper_aux_size`` because the
    default Aux size used by the checkpointed counter must scale with this
    value.  For n=256 the worst case is 19 scratch qubits plus the temporary
    Ctrl bit, i.e. Aux=20.  For n=512 the unary path depth increases by one
    on each of the two endpoint scans, so the worst-case scratch is 21 and
    Aux must be 22.
    """
    if k > K:
        return 0
    endpoint_width = max(len_width, shift_width)
    rel_count = K - k + 1
    labels_main = list(range(rel_count))
    if rel_count > 1 and ((rel_count - 1) & (rel_count - 2)) == 0:
        # Same top-special split as lc_interval_addsub_unary_gate.
        labels_main = list(range(rel_count - 1))
    depth = _tight_unary_depth_for_labels(labels_main) if labels_main else 0
    base = max(2 * depth, endpoint_width)
    return base + 3


def _fastdual_prefix_scratch_size(k: int, K: int, len_width: int) -> int:
    if k > K:
        return 0
    depth = _e.unary_depth(K - k + 1)
    return max(depth, len_width) + 3


def _fastdual_interval_scratch_size(label_count: int, endpoint_width: int) -> int:
    """Scratch qubits used by lc_interval_addsub_unary_gate.

    The FASTDUAL interval Add/Sub block handles a one-more-than-a-power-of-two
    interval by pulling the top label out as a special endpoint.  Its two endpoint
    unary paths therefore have depth based on ``main_count`` rather than directly
    on ``label_count``.  The scratch layout in lc_interval_addsub_unary_gate is

        base = max(2*depth, endpoint_width)
        Scratch[base], Scratch[base+1], Scratch[base+2]

    so the number of scratch qubits needed by the block is ``base + 3``.
    This is 19 for n=256 but grows to 21 for n=384/512; the previous hard-coded
    lower bound of 19 caused the n=512 qubit-arity mismatch.
    """
    if label_count <= 1:
        depth = 0
    else:
        main_count = label_count
        if label_count > 1 and ((label_count - 1) & (label_count - 2)) == 0:
            main_count = label_count - 1
        depth = 0 if main_count <= 1 else (main_count - 1).bit_length()
    return max(2 * depth, endpoint_width) + 3


def qiskit_paper_aux_size(n: int, len_width: int, shift_width: int, T_max: Optional[int] = None,
                          include_algorithm1: bool = False) -> int:
    # Includes the temporary Ctrl bit Aux[0].  For n=256 this is exactly 20;
    # for larger n the FASTDUAL r-side interval Add/Sub can require more.
    if T_max is None:
        T_max = _e.Nmax_steps(n)
    max_swap = max_t = max_l4 = max_l5 = 1
    max_r_interval_scratch = 1
    endpoint_width = max(len_width, shift_width)
    for T in range(1, T_max + 1):
        w = _e.active_windows(n, T)
        r_count = w["r_addsub"][1] - w["r_addsub"][0] + 1
        max_r_interval_scratch = max(max_r_interval_scratch, _fastdual_interval_scratch_size(r_count, endpoint_width))
        max_swap = max(max_swap, w["swap"][1] - w["swap"][0] + 1)
        max_t = max(max_t, w["t_addsub"][1] - w["t_addsub"][0] + 1)
        max_l4 = max(max_l4, w["len_update_lt"][1] - w["len_update_lt"][0] + 1)
        max_l5 = max(max_l5, w["len_update_lrp"][1] - w["len_update_lrp"][0] + 1)
    step_scratch = max(
        shift_width + 4,
        max_r_interval_scratch,
        max(len_width + 1, _e.unary_depth(max_swap)),
        max(_e.unary_depth(max_t) + 3, len_width + 1),
        max(len_width, shift_width) + 3,
        max(len_width + 1, _e.unary_depth(max(max_l4, max_l5)) + 2),
        len_width - 1 + 2,
        max(len_width, shift_width) + 6,
    )
    return max(1 + step_scratch, 20)  # includes temporary Ctrl

def make_global_registers_noctrl(*, n: int, len_width: int, shift_width: int,
                                 T_max: Optional[int] = None, include_algorithm1: bool = False,
                                 aux_size: Optional[int] = None):
    work_size = n + 3
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
    if aux_size is None:
        aux_size = qiskit_paper_aux_size(n, len_width, shift_width, T_max, include_algorithm1)
    Aux = QuantumRegister(aux_size, "Aux")
    return Phase1, Phase2, Iter, Sign, Work1, Work2, l_t, l_q, l_s, l_rp, Aux


def _make_condition(qc: QuantumCircuit, conditions, out: Qubit, scratch: Sequence[Qubit]) -> None:
    _e.compute_control(qc, conditions, out, scratch)


def append_one_step_T(qc: QuantumCircuit, *, T: int, n: int, len_width: int, shift_width: int,
                      Phase1, Phase2, Iter, Sign, Work1, Work2, l_t, l_q, l_s, l_rp, Aux) -> None:
    work_size = n + 3
    windows = _e.active_windows(n, T)
    k1, K1 = windows["r_addsub"]
    k2, K2 = windows["swap"]
    k3, K3 = windows["t_addsub"]
    k4, K4 = windows["len_update_lt"]
    k5, K5 = windows["len_update_lrp"]
    ctrl = Aux[0]
    scratch = list(Aux[1:])
    pool = scratch
    # Pre-shift
    pre = _e.pre_shift_gate(work_size=work_size, shift_width=shift_width)
    _e._append_with_optional_clbits(qc, pre, [Phase1[0], Phase2[0]] + list(Work2) + list(l_s) + scratch[:pre.num_qubits-(2+work_size+shift_width)])
    # R sub: Phase1=0
    _make_condition(qc, [(Phase1[0], 0)], ctrl, scratch)
    rsub = lc_interval_addsub_unary_gate(n=n, k=k1, K=K1, len_width=len_width, shift_width=shift_width,
                                         mode="sub", sign_update=True, target="work1", name="R_SUB_S835_FAST")
    _e._append_with_optional_clbits(qc, rsub, [ctrl, Sign[0]] + list(Work1[k1-1:K1]) + list(Work2[k1-1:K1])
                                    + list(l_t) + list(l_q) + list(l_s) + scratch[:rsub.num_qubits-(2+2*(K1-k1+1)+len_width+len_width+shift_width)])
    _make_condition(qc, [(Phase1[0], 0)], ctrl, scratch)
    # if Phase1=0 and Phase2=1 then Sign ^= 1
    _make_condition(qc, [(Phase1[0], 0), (Phase2[0], 1)], ctrl, scratch)
    qc.cx(ctrl, Sign[0])
    _make_condition(qc, [(Phase1[0], 0), (Phase2[0], 1)], ctrl, scratch)
    # R add: Phase1=0 and not(Phase2&Sign). Use scratch[0] as tmp, uncomputed during block.
    tmp = scratch[0]
    qc.ccx(Phase2[0], Sign[0], tmp)
    _make_condition(qc, [(Phase1[0], 0), (tmp, 0)], ctrl, scratch[1:])
    qc.ccx(Phase2[0], Sign[0], tmp)
    radd = lc_interval_addsub_unary_gate(n=n, k=k1, K=K1, len_width=len_width, shift_width=shift_width,
                                         mode="add", sign_update=False, target="work1", name="R_ADD_S835_FAST")
    _e._append_with_optional_clbits(qc, radd, [ctrl, Sign[0]] + list(Work1[k1-1:K1]) + list(Work2[k1-1:K1])
                                    + list(l_t) + list(l_q) + list(l_s) + scratch[:radd.num_qubits-(2+2*(K1-k1+1)+len_width+len_width+shift_width)])
    qc.ccx(Phase2[0], Sign[0], tmp)
    _make_condition(qc, [(Phase1[0], 0), (tmp, 0)], ctrl, scratch[1:])
    qc.ccx(Phase2[0], Sign[0], tmp)
    # Swap: ctrl = Phase1 xor Phase2
    qc.cx(Phase1[0], ctrl); qc.cx(Phase2[0], ctrl)
    lcs = lc_swap_unary_gate(k=k2, K=K2, len_width=len_width)
    _e._append_with_optional_clbits(qc, lcs, [ctrl, Sign[0]] + list(Work1[k2-1:K2]) + list(l_t) + list(l_q)
                                    + scratch[:lcs.num_qubits-(2+(K2-k2+1)+len_width+len_width)])
    qc.cx(Phase2[0], ctrl); qc.cx(Phase1[0], ctrl)
    # l_q +/- updates.
    _make_condition(qc, [(Phase1[0], 1), (Phase2[0], 0)], ctrl, scratch)
    _e.dec_mod2n_1ctrl(qc, ctrl, list(l_q), scratch[:max(0,len_width-1)])
    _make_condition(qc, [(Phase1[0], 1), (Phase2[0], 0)], ctrl, scratch)
    _make_condition(qc, [(Phase1[0], 0), (Phase2[0], 1)], ctrl, scratch)
    _e.inc_mod2n_1ctrl(qc, ctrl, list(l_q), scratch[:max(0,len_width-1)])
    _make_condition(qc, [(Phase1[0], 0), (Phase2[0], 1)], ctrl, scratch)
    # T sub condition: Phase1=1 and (Phase2=1 or Sign=0)
    tmp = scratch[0]
    _make_condition(qc, [(Phase2[0], 0), (Sign[0], 1)], tmp, scratch[1:])
    _make_condition(qc, [(Phase1[0], 1), (tmp, 0)], ctrl, scratch[1:])
    _make_condition(qc, [(Phase2[0], 0), (Sign[0], 1)], tmp, scratch[1:])
    tsub = lc_prefix_addsub_unary_gate(k=k3, K=K3, len_width=len_width,
                                       mode="sub", sign_update=False, target="work2", name="T_SUB_S835_FAST")
    _e._append_with_optional_clbits(qc, tsub, [ctrl, Sign[0]] + list(Work1[k3-1:K3]) + list(Work2[k3-1:K3])
                                    + list(l_t) + scratch[:tsub.num_qubits-(2+2*(K3-k3+1)+len_width)])
    _make_condition(qc, [(Phase2[0], 0), (Sign[0], 1)], tmp, scratch[1:])
    _make_condition(qc, [(Phase1[0], 1), (tmp, 0)], ctrl, scratch[1:])
    _make_condition(qc, [(Phase2[0], 0), (Sign[0], 1)], tmp, scratch[1:])
    qc.cx(Phase1[0], Sign[0])
    _make_condition(qc, [(Phase1[0], 1)], ctrl, scratch)
    tadd = lc_prefix_addsub_unary_gate(k=k3, K=K3, len_width=len_width,
                                       mode="add", sign_update=True, target="work2", name="T_ADD_S835_FAST")
    _e._append_with_optional_clbits(qc, tadd, [ctrl, Sign[0]] + list(Work1[k3-1:K3]) + list(Work2[k3-1:K3])
                                    + list(l_t) + scratch[:tadd.num_qubits-(2+2*(K3-k3+1)+len_width)])
    _make_condition(qc, [(Phase1[0], 1)], ctrl, scratch)
    # Post-shift
    post = _e.post_shift_gate(work_size=work_size, shift_width=shift_width)
    _e._append_with_optional_clbits(qc, post, [Phase1[0], Phase2[0]] + list(Work2) + list(l_s) + scratch[:post.num_qubits-(2+work_size+shift_width)])
    # Phase update
    pupdate = _e.phase_update_gate(len_width=len_width, shift_width=shift_width)
    _e._append_with_optional_clbits(qc, pupdate, [Phase1[0], Phase2[0], Sign[0]] + list(l_q) + list(l_rp) + list(l_s)
                                    + scratch[:pupdate.num_qubits-(3+len_width+len_width+shift_width)])
    # End iteration every four steps.
    if T % 4 == 0:
        z_lq = scratch[0]; z_ls = scratch[1]; eq_pool = scratch[2:]
        _e.mcx_vchain(qc, list(l_q), z_lq, eq_pool)
        _e.mcx_vchain(qc, list(l_s), z_ls, eq_pool)
        qc.ccx(z_lq, z_ls, ctrl)
        swlen = swap_work_and_len_unary_shared_gate(n=n, len_width=len_width, k4=k4, K4=K4, k5=k5, K5=K5)
        need = swlen.num_qubits - (1+2*work_size+2*len_width)
        _e._append_with_optional_clbits(qc, swlen, [ctrl] + list(Work1) + list(Work2) + list(l_t) + list(l_rp) + scratch[2:2+need])
        qc.cx(ctrl, Iter[0])
        qc.ccx(z_lq, z_ls, ctrl)
        _e.mcx_vchain(qc, list(l_s), z_ls, eq_pool)
        _e.mcx_vchain(qc, list(l_q), z_lq, eq_pool)


def build_step_circuit(n:int, T:int, *, T_max:Optional[int]=None, aux_size:Optional[int]=None, measurement_uncompute:bool=True):
    cfg=get_n_config(n); lw=int(cfg['len_width']); sw=int(cfg['shift_width']); T_max=int(T_max or cfg['T_max'])
    if aux_size is None: aux_size=qiskit_paper_aux_size(n,lw,sw,T_max)
    set_measurement_uncompute(measurement_uncompute)
    regs=make_global_registers_noctrl(n=n,len_width=lw,shift_width=sw,T_max=T_max,aux_size=aux_size)
    qc=QuantumCircuit(*regs, name=f"S835_FASTDUAL_STEP_T{T}_{n}")
    Phase1,Phase2,Iter,Sign,Work1,Work2,l_t,l_q,l_s,l_rp,Aux=regs
    append_one_step_T(qc,T=T,n=n,len_width=lw,shift_width=sw,Phase1=Phase1,Phase2=Phase2,Iter=Iter,Sign=Sign,Work1=Work1,Work2=Work2,l_t=l_t,l_q=l_q,l_s=l_s,l_rp=l_rp,Aux=Aux)
    return qc

if __name__ == '__main__':
    import argparse,json
    ap=argparse.ArgumentParser(); ap.add_argument('--n',type=int,default=256); ap.add_argument('--T',type=int,default=1); ap.add_argument('--count',action='store_true'); args=ap.parse_args()
    cfg=get_n_config(args.n); lw=int(cfg['len_width']); sw=int(cfg['shift_width']); Tm=int(cfg['T_max'])
    out={'n':args.n,'len_width':lw,'shift_width':sw,'T_max':Tm,'aux_size':qiskit_paper_aux_size(args.n,lw,sw,Tm)}
    qc=build_step_circuit(args.n,args.T,T_max=Tm)
    out['step_qubits']=qc.num_qubits; out['top_ops']={str(k):int(v) for k,v in qc.count_ops().items()}
    if args.count:
        out['ops']={str(k):int(v) for k,v in _e.count_circuit_ops_recursive(qc).items()}
    print(json.dumps(out,indent=2,sort_keys=True))
