import argparse
import inspect
import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import eea_circuit as eea
except ImportError as exc:
    raise SystemExit(
        "Cannot import eea_circuit.py. Put this test next to the circuit file, "
        "or change `import eea_circuit as eea` at the top of this test.\n"
        f"Original import error: {exc}"
    )

try:
    from qiskit import QuantumCircuit, QuantumRegister
except Exception as exc:
    raise SystemExit(
        "This strict test requires Qiskit because it builds the PDF block gates.\n"
        f"Original import error: {exc}"
    )

PRIMITIVES = {"x", "cx", "cnot", "ccx", "tof", "toffoli"}
IGNORED = {"barrier", "delay", "id"}


# Small basis-state simulator for recursively expanded X/CX/CCX circuits.
def _qindex(circuit, qubit) -> int:
    return circuit.find_bit(qubit).index


def _get_qreg(circuit, name: str):
    for reg in circuit.qregs:
        if reg.name == name:
            return reg
    raise KeyError(f"No QuantumRegister named {name!r}; available: {[r.name for r in circuit.qregs]}")


def _iter_items(circuit):
    for item in circuit.data:
        yield item.operation, item.qubits


def _apply_instruction(inst, global_qids: Sequence[int], bits: List[int]) -> None:
    name = inst.name.lower()

    if name == "x":
        bits[global_qids[0]] ^= 1
        return
    if name in {"cx", "cnot"}:
        if bits[global_qids[0]]:
            bits[global_qids[1]] ^= 1
        return
    if name in {"ccx", "tof", "toffoli"}:
        if bits[global_qids[0]] and bits[global_qids[1]]:
            bits[global_qids[2]] ^= 1
        return
    if name in IGNORED:
        return

    definition = getattr(inst, "definition", None)
    if definition is None:
        raise ValueError(
            f"Unsupported non-Toffoli leaf instruction {inst.name!r}; "
            "the PDF construction should expand to X/CX/CCX only."
        )

    for subinst, subqubits in _iter_items(definition):
        sub_global_qids = [global_qids[_qindex(definition, q)] for q in subqubits]
        _apply_instruction(subinst, sub_global_qids, bits)


def simulate_basis(circuit, initial_bits: Dict[int, int]) -> List[int]:
    bits = [0] * circuit.num_qubits
    for qid, value in initial_bits.items():
        bits[qid] = int(value) & 1
    for inst, qargs in _iter_items(circuit):
        global_qids = [_qindex(circuit, q) for q in qargs]
        _apply_instruction(inst, global_qids, bits)
    return bits


def collect_nonprimitive_leaves(obj, *, limit: int = 20) -> List[str]:
    """Return non-X/CX/CCX leaf instructions appearing recursively."""
    bad: List[str] = []

    def walk_circuit(circ) -> None:
        for inst, _qargs in _iter_items(circ):
            walk_inst(inst)
            if len(bad) >= limit:
                return

    def walk_inst(inst) -> None:
        name = inst.name.lower()
        if name in PRIMITIVES or name in IGNORED:
            return
        definition = getattr(inst, "definition", None)
        if definition is None:
            bad.append(inst.name)
            return
        walk_circuit(definition)

    if hasattr(obj, "data"):
        walk_circuit(obj)
    else:
        walk_inst(obj)
    return bad[:limit]


def assert_toffoli_network(obj, label: str) -> None:
    bad = collect_nonprimitive_leaves(obj)
    if bad:
        raise AssertionError(
            f"{label} does not recursively expand to an X/CX/CCX Toffoli network. "
            f"First unsupported leaves: {bad}"
        )


# Register encoding helpers.
def encoded_zero(width: int) -> int:
    return (1 << width) - 1


def enc_len(actual_length: int, width: int) -> int:
    if actual_length < 0:
        raise ValueError("length must be non-negative")
    return encoded_zero(width) if actual_length == 0 else (actual_length - 1) % (1 << width)


def dec_len(encoded_value: int, width: int) -> int:
    return 0 if encoded_value == encoded_zero(width) else encoded_value + 1


def set_reg_int_le(init: Dict[int, int], circuit, reg, value: int, width: Optional[int] = None) -> None:
    if width is None:
        width = len(reg)
    value %= 1 << width
    for i in range(width):
        if (value >> i) & 1:
            init[_qindex(circuit, reg[i])] = 1


def get_reg_int_le(bits: Sequence[int], circuit, reg, start: int = 0, width: Optional[int] = None) -> int:
    if width is None:
        width = len(reg) - start
    out = 0
    for i in range(width):
        out |= bits[_qindex(circuit, reg[start + i])] << i
    return out


def set_reg_int_be(init: Dict[int, int], circuit, reg, start: int, width: int, value: int) -> None:
    for j in range(width):
        if (value >> (width - 1 - j)) & 1:
            init[_qindex(circuit, reg[start + j])] = 1


def get_reg_bits_lr(bits: Sequence[int], circuit, reg) -> str:
    return "".join(str(bits[_qindex(circuit, q)]) for q in reg)


def set_bits_lr(init: Dict[int, int], circuit, reg, bit_string: str) -> None:
    if len(bit_string) != len(reg):
        raise ValueError(f"bit string length {len(bit_string)} != register width {len(reg)}")
    for i, b in enumerate(bit_string):
        if b == "1":
            init[_qindex(circuit, reg[i])] = 1
        elif b != "0":
            raise ValueError(f"invalid bit {b!r}")


def clean_reg(bits: Sequence[int], circuit, reg) -> bool:
    return all(bits[_qindex(circuit, q)] == 0 for q in reg)


# Classical endpoint oracle for Algorithm 3.
@dataclass(frozen=True)
class ClassicalExpected:
    x_used: int
    iter_start: int
    iter_final: int
    tprime_final: int
    exact_step_count: int
    inverse: int


def classical_expected_for_algorithm3(p: int, x: int) -> ClassicalExpected:
    if not (1 <= x < p):
        raise ValueError(f"x must satisfy 1 <= x < p; got x={x}, p={p}")
    if math.gcd(x, p) != 1:
        raise ValueError(f"x={x} is not invertible modulo p={p}")

    iter_bit = 0
    x_used = x
    if x > p // 2:
        iter_bit = 1
        x_used = p - x

    r_prev, r = p, x_used
    t_prev, t = 0, 1
    exact_steps = 0

    while r != 0:
        q = r_prev // r
        exact_steps += 4 * q.bit_length()
        r_prev, r = r, r_prev - q * r
        t_prev, t = t, t_prev + q * t
        iter_bit ^= 1

    return ClassicalExpected(
        x_used=x_used,
        iter_start=1 if x > p // 2 else 0,
        iter_final=iter_bit,
        tprime_final=t_prev % p,
        exact_step_count=exact_steps,
        inverse=pow(x, -1, p),
    )


def initialize_algorithm3_input_bits(circuit, *, p: int, x: int, n: int, len_width: int, shift_width: int):
    expected = classical_expected_for_algorithm3(p, x)

    Work1 = _get_qreg(circuit, "Work1")
    Work2 = _get_qreg(circuit, "Work2")
    Iter = _get_qreg(circuit, "Iter")
    l_q = _get_qreg(circuit, "l_q")
    l_s = _get_qreg(circuit, "l_s")
    l_rp = _get_qreg(circuit, "l_rp")

    init: Dict[int, int] = {}
    init[_qindex(circuit, Work1[0])] = 1
    set_reg_int_be(init, circuit, Work1, start=3, width=n, value=p)
    set_reg_int_be(init, circuit, Work2, start=3, width=n, value=expected.x_used)

    set_reg_int_le(init, circuit, l_q, encoded_zero(len_width))
    set_reg_int_le(init, circuit, l_s, encoded_zero(shift_width))
    set_reg_int_le(init, circuit, l_rp, expected.x_used.bit_length() - 1)

    if expected.iter_start:
        init[_qindex(circuit, Iter[0])] = 1
    return init, expected


@dataclass
class EndpointResult:
    p: int
    x: int
    label: str
    T: int
    exact_T: int
    tprime: int
    exp_tprime: int
    iter_bit: int
    exp_iter: int
    inverse: int
    exp_inverse: int
    l_q_actual: int
    l_s_actual: int
    l_rp_actual: int
    aux_clean: bool
    ctrl_clean: bool
    work1_bits: str
    work2_bits: str


def run_algorithm3_endpoint_case(*, p: int, x: int, T: int, label: str) -> EndpointResult:
    n = p.bit_length()
    cfg = eea.get_n_config(n)
    len_width = cfg["len_width"]
    shift_width = cfg["shift_width"]

    circuit = eea.build_full_steps_circuit(n=n, len_width=len_width, shift_width=shift_width, T_max=T)
    assert_toffoli_network(circuit, f"build_full_steps_circuit(n={n}, T={T})")
    init, expected = initialize_algorithm3_input_bits(
        circuit, p=p, x=x, n=n, len_width=len_width, shift_width=shift_width
    )
    final = simulate_basis(circuit, init)

    Iter = _get_qreg(circuit, "Iter")
    Ctrl = _get_qreg(circuit, "Ctrl")
    Work1 = _get_qreg(circuit, "Work1")
    Work2 = _get_qreg(circuit, "Work2")
    l_q = _get_qreg(circuit, "l_q")
    l_s = _get_qreg(circuit, "l_s")
    l_rp = _get_qreg(circuit, "l_rp")
    Aux = _get_qreg(circuit, "Aux")

    tprime = get_reg_int_le(final, circuit, Work2, start=0, width=n) % p
    iter_bit = final[_qindex(circuit, Iter[0])]
    inverse = tprime if iter_bit else (-tprime % p)

    return EndpointResult(
        p=p,
        x=x,
        label=label,
        T=T,
        exact_T=expected.exact_step_count,
        tprime=tprime,
        exp_tprime=expected.tprime_final,
        iter_bit=iter_bit,
        exp_iter=expected.iter_final,
        inverse=inverse,
        exp_inverse=expected.inverse,
        l_q_actual=dec_len(get_reg_int_le(final, circuit, l_q), len_width),
        l_s_actual=dec_len(get_reg_int_le(final, circuit, l_s), shift_width),
        l_rp_actual=dec_len(get_reg_int_le(final, circuit, l_rp), len_width),
        aux_clean=clean_reg(final, circuit, Aux),
        ctrl_clean=(final[_qindex(circuit, Ctrl[0])] == 0),
        work1_bits=get_reg_bits_lr(final, circuit, Work1),
        work2_bits=get_reg_bits_lr(final, circuit, Work2),
    )


def check_endpoint_result(r: EndpointResult) -> List[str]:
    reasons: List[str] = []
    if r.tprime != r.exp_tprime:
        reasons.append(f"t' got {r.tprime}, expected {r.exp_tprime}")
    if r.iter_bit != r.exp_iter:
        reasons.append(f"Iter got {r.iter_bit}, expected {r.exp_iter}")
    if r.inverse != r.exp_inverse:
        reasons.append(f"inverse got {r.inverse}, expected {r.exp_inverse}")
    if r.l_rp_actual != 0:
        reasons.append(f"ell_r' actual length got {r.l_rp_actual}, expected 0")
    if not r.aux_clean:
        reasons.append("Aux is not clean")
    if not r.ctrl_clean:
        reasons.append("Ctrl is not clean")
    return reasons


# ---------------------------------------------------------------------------
# Structural tests that reject endpoint shortcuts.
# ---------------------------------------------------------------------------

def top_level_names(circuit) -> List[str]:
    return [inst.name for inst, _qargs in _iter_items(circuit)]


def test_no_small_reference_shortcut() -> None:
    src = inspect.getsource(eea.build_full_steps_circuit)
    bad_tokens = [
        "_build_small_algorithm3_reference_circuit",
        "SMALL_REFERENCE",
        "small reference endpoint",
        "n <= 4",
    ]
    found = [tok for tok in bad_tokens if tok in src]
    if found:
        raise AssertionError(
            "build_full_steps_circuit appears to contain a small-n reference/endpoint shortcut. "
            f"Forbidden source tokens found: {found}"
        )

    n = 4
    cfg = eea.get_n_config(n)
    qc = eea.build_full_steps_circuit(n=n, len_width=cfg["len_width"], shift_width=cfg["shift_width"], T_max=4)
    if "REFERENCE" in qc.name.upper() or len(qc.data) == 0:
        raise AssertionError(
            f"n={n} step circuit must be composed from Algorithm-3 blocks; got name={qc.name!r}, "
            f"top-level ops={len(qc.data)}"
        )


def test_step_schedule_structure() -> None:
    n = 5
    cfg = eea.get_n_config(n)
    qc = eea.build_full_steps_circuit(n=n, len_width=cfg["len_width"], shift_width=cfg["shift_width"], T_max=4)
    names = top_level_names(qc)
    required_substrings = [
        "PRE_SHIFT",
        "R_SUB",
        "R_ADD",
        "LC_SWAP",
        "T_SUB",
        "T_ADD",
        "POST_SHIFT",
        "PHASE_UPDATE",
        "SWAP_AND_LEN",
    ]
    missing = [needle for needle in required_substrings if not any(needle in name for name in names)]
    if missing:
        raise AssertionError(
            "The step circuit does not expose the expected PDF Algorithm-3 dashed-block schedule. "
            f"Missing top-level block names containing: {missing}. First names: {names[:80]}"
        )
    assert_toffoli_network(qc, "Algorithm-3 step circuit structure")


def expected_active_windows(n: int, T: int) -> Dict[str, Tuple[int, int]]:
    c = 1.0 / math.log2((math.sqrt(5.0) + 1.0) / 2.0)
    ceil = lambda x: math.ceil(x - 1e-12)
    floor = lambda x: math.floor(x + 1e-12)
    return {
        "r_addsub": (max(ceil((T - (n + 2)) / (4.0 * c - 1.0)), 1) + 2, n + 3),
        "swap": (max(ceil((T - 3.0 * (n + 2)) / (4.0 * c - 3.0)), 1) + 1,
                  min(floor(T / 2.0) + 2, n + 2)),
        "t_addsub": (1, min(ceil(T / 4.0) + 1, n + 1)),
        "len_update_lt": (max(ceil((T - 4.0 * (n + 2)) / (4.0 * c - 4.0)), 1),
                          min(floor(T / 4.0 + 3.0), n + 3)),
        "len_update_lrp": (ceil(T / (4.0 * c)), min(floor(T / 4.0 + 4.0), n + 3)),
    }


def test_active_window_formulas() -> None:
    for n in [4, 5, 6, 8, 16]:
        T_max = min(eea.Nmax_steps(n), 4 * n + 8)
        for T in range(1, T_max + 1):
            got = eea.active_windows(n, T)
            exp = expected_active_windows(n, T)
            if got != exp:
                raise AssertionError(f"active_windows mismatch for n={n}, T={T}: got {got}, expected {exp}")


# Block-level unit tests.
def build_gate_circuit(gate, *regs):
    qc = QuantumCircuit(*regs, name=f"TEST_{gate.name}")
    qc.append(gate, [q for reg in regs for q in reg])
    assert_toffoli_network(qc, qc.name)
    return qc


def rotate_left(bits: str, amount: int) -> str:
    amount %= len(bits)
    return bits[amount:] + bits[:amount]


def rotate_right(bits: str, amount: int) -> str:
    amount %= len(bits)
    return bits[-amount:] + bits[:-amount] if amount else bits


def test_pre_post_shift_blocks() -> None:
    work_size = 6
    shift_width = 4
    work_patterns = ["100101", "011010"]
    for gate_name, gate_factory in [
        ("pre", eea.pre_shift_gate),
        ("post", eea.post_shift_gate),
    ]:
        gate = gate_factory(work_size=work_size, shift_width=shift_width)
        Phase1 = QuantumRegister(1, "Phase1")
        Phase2 = QuantumRegister(1, "Phase2")
        Work2 = QuantumRegister(work_size, "Work2")
        l_s = QuantumRegister(shift_width, "l_s")
        Scratch = QuantumRegister(gate.num_qubits - (2 + work_size + shift_width), "Scratch")
        qc = build_gate_circuit(gate, Phase1, Phase2, Work2, l_s, Scratch)

        for p1 in [0, 1]:
            for p2 in [0, 1]:
                for ls_actual in [0, 1, 2, 3]:
                    for pattern in work_patterns:
                        init: Dict[int, int] = {}
                        if p1:
                            init[_qindex(qc, Phase1[0])] = 1
                        if p2:
                            init[_qindex(qc, Phase2[0])] = 1
                        set_bits_lr(init, qc, Work2, pattern)
                        set_reg_int_le(init, qc, l_s, enc_len(ls_actual, shift_width))
                        out = simulate_basis(qc, init)

                        exp_pattern = pattern
                        exp_enc = enc_len(ls_actual, shift_width)
                        active = (p1 == 0) if gate_name == "pre" else (p1 == 1)
                        if active:
                            exp_pattern = rotate_left(exp_pattern, 1)
                            exp_enc = (exp_enc + 1) % (1 << shift_width)
                            if p2 == 1:
                                exp_pattern = rotate_right(exp_pattern, 2)
                                exp_enc = (exp_enc - 2) % (1 << shift_width)

                        got_pattern = get_reg_bits_lr(out, qc, Work2)
                        got_enc = get_reg_int_le(out, qc, l_s)
                        if got_pattern != exp_pattern or got_enc != exp_enc or not clean_reg(out, qc, Scratch):
                            raise AssertionError(
                                f"{gate_name}_shift mismatch p1={p1}, p2={p2}, ls={ls_actual}, pattern={pattern}: "
                                f"got pattern={got_pattern}, enc_l_s={got_enc}, scratch_clean={clean_reg(out, qc, Scratch)}; "
                                f"expected pattern={exp_pattern}, enc_l_s={exp_enc}"
                            )


def test_phase_update_block() -> None:
    len_width = 3
    shift_width = 4
    gate = eea.phase_update_gate(len_width=len_width, shift_width=shift_width)
    Phase1 = QuantumRegister(1, "Phase1")
    Phase2 = QuantumRegister(1, "Phase2")
    Sign = QuantumRegister(1, "Sign")
    l_q = QuantumRegister(len_width, "l_q")
    l_rp = QuantumRegister(len_width, "l_rp")
    l_s = QuantumRegister(shift_width, "l_s")
    Scratch = QuantumRegister(gate.num_qubits - (3 + 2 * len_width + shift_width), "Scratch")
    qc = build_gate_circuit(gate, Phase1, Phase2, Sign, l_q, l_rp, l_s, Scratch)

    for p1 in [0, 1]:
        for p2 in [0, 1]:
            for sign in [0, 1]:
                for ell_q in [0, 1, 2]:
                    for ell_rp in [0, 1, 2]:
                        for ell_s in [0, 1, 2]:
                            init: Dict[int, int] = {}
                            if p1:
                                init[_qindex(qc, Phase1[0])] = 1
                            if p2:
                                init[_qindex(qc, Phase2[0])] = 1
                            if sign:
                                init[_qindex(qc, Sign[0])] = 1
                            set_reg_int_le(init, qc, l_q, enc_len(ell_q, len_width))
                            set_reg_int_le(init, qc, l_rp, enc_len(ell_rp, len_width))
                            set_reg_int_le(init, qc, l_s, enc_len(ell_s, shift_width))

                            out = simulate_basis(qc, init)
                            ep1, ep2, esign = p1, p2, sign
                            if ell_q == 0 and ell_rp > 0:
                                ep2 = ep2 ^ esign ^ ep1
                                esign = esign ^ ep2  # Algorithm/circuit order uses updated Phase2.
                            if ell_s == 0:
                                ep1 ^= 1
                                ep2 ^= 1

                            got = (
                                out[_qindex(qc, Phase1[0])],
                                out[_qindex(qc, Phase2[0])],
                                out[_qindex(qc, Sign[0])],
                            )
                            exp = (ep1, ep2, esign)
                            if got != exp or not clean_reg(out, qc, Scratch):
                                raise AssertionError(
                                    f"phase_update mismatch input p1={p1},p2={p2},sign={sign},"
                                    f"ell_q={ell_q},ell_rp={ell_rp},ell_s={ell_s}: got {got}, expected {exp}"
                                )


def test_unary_iteration_selector() -> None:
    labels = [5, 6, 7, 8]
    width = 4
    selected = 7
    depth = eea.unary_depth(len(labels))
    Ctrl = QuantumRegister(1, "Ctrl")
    A = QuantumRegister(width, "A")
    Target = QuantumRegister(1, "Target")
    Anc = QuantumRegister(depth, "Anc")
    qc = QuantumCircuit(Ctrl, A, Target, Anc, name="TEST_UNARY_ITERATION")

    def leaf(j: int, ej) -> None:
        if j == selected:
            qc.cx(ej, Target[0])

    eea.unary_iteration(qc, index_reg=A, labels=labels, ctrl=Ctrl[0], ancillas=Anc, leaf_fn=leaf)
    assert_toffoli_network(qc, "unary_iteration selector")

    for ctrl in [0, 1]:
        for a in labels:
            for target in [0, 1]:
                init: Dict[int, int] = {}
                if ctrl:
                    init[_qindex(qc, Ctrl[0])] = 1
                if target:
                    init[_qindex(qc, Target[0])] = 1
                set_reg_int_le(init, qc, A, a)
                out = simulate_basis(qc, init)
                exp = target ^ (1 if (ctrl and a == selected) else 0)
                got = out[_qindex(qc, Target[0])]
                if got != exp or not clean_reg(out, qc, Anc):
                    raise AssertionError(
                        f"unary_iteration selector mismatch ctrl={ctrl}, A={a}, target={target}: "
                        f"got target={got}, expected {exp}, anc_clean={clean_reg(out, qc, Anc)}"
                    )


def test_lc_swap_unary_block() -> None:
    k, K = 2, 5
    len_width = 3
    gate = eea.lc_swap_unary_gate(k=k, K=K, len_width=len_width)
    M = K - k + 1
    Ctrl = QuantumRegister(1, "Ctrl")
    Sign = QuantumRegister(1, "Sign")
    Work1 = QuantumRegister(M, "Work1")
    l_t = QuantumRegister(len_width, "l_t")
    l_q = QuantumRegister(len_width, "l_q")
    Scratch = QuantumRegister(gate.num_qubits - (2 + M + 2 * len_width), "Scratch")
    qc = build_gate_circuit(gate, Ctrl, Sign, Work1, l_t, l_q, Scratch)

    pattern = "1010"
    for ctrl in [0, 1]:
        for sign in [0, 1]:
            for J in range(k, K + 1):
                ell_t = 1
                ell_q = J - ell_t - 1
                init: Dict[int, int] = {}
                if ctrl:
                    init[_qindex(qc, Ctrl[0])] = 1
                if sign:
                    init[_qindex(qc, Sign[0])] = 1
                set_bits_lr(init, qc, Work1, pattern)
                set_reg_int_le(init, qc, l_t, enc_len(ell_t, len_width))
                set_reg_int_le(init, qc, l_q, enc_len(ell_q, len_width))
                out = simulate_basis(qc, init)

                exp_bits = list(pattern)
                exp_sign = sign
                if ctrl:
                    idx = J - k
                    exp_bits[idx], exp_sign = str(sign), int(exp_bits[idx])
                got_bits = get_reg_bits_lr(out, qc, Work1)
                got_sign = out[_qindex(qc, Sign[0])]
                if (
                    got_bits != "".join(exp_bits)
                    or got_sign != exp_sign
                    or get_reg_int_le(out, qc, l_t) != enc_len(ell_t, len_width)
                    or get_reg_int_le(out, qc, l_q) != enc_len(ell_q, len_width)
                    or not clean_reg(out, qc, Scratch)
                ):
                    raise AssertionError(
                        f"lc_swap mismatch ctrl={ctrl}, sign={sign}, J={J}: "
                        f"got Sign={got_sign}, Work1={got_bits}, scratch_clean={clean_reg(out, qc, Scratch)}; "
                        f"expected Sign={exp_sign}, Work1={''.join(exp_bits)}"
                    )


def highest_position_encoded(bits_lr: str, *, k: int, K: int, B: int, len_width: int) -> int:
    mask = encoded_zero(len_width)
    for pos in range(K, k - 1, -1):
        if pos <= B and bits_lr[pos - k] == "1":
            return (pos - 1) & mask
    return mask


def right_length_encoded(bits_lr: str, *, n: int, k: int, K: int, A: int, len_width: int) -> int:
    mask = encoded_zero(len_width)
    for pos in range(k, K + 1):
        if pos >= A and bits_lr[pos - k] == "1":
            return (n + 3 - pos) & mask
    return mask


def test_length_update_blocks() -> None:
    n = 4
    len_width = 3

    # Test len_update_lt: l_t ^= highest(Work2 valid) ^ highest(Work1 valid), encoded as len-1.
    k, K = 1, 4
    gate = eea.len_update_lt_unary_gate(n=n, k=k, K=K, len_width=len_width)
    M = K - k + 1
    Ctrl = QuantumRegister(1, "Ctrl")
    Work1 = QuantumRegister(M, "Work1")
    Work2 = QuantumRegister(M, "Work2")
    l_t = QuantumRegister(len_width, "l_t")
    l_rp = QuantumRegister(len_width, "l_rp")
    Scratch = QuantumRegister(gate.num_qubits - (1 + 2 * M + 2 * len_width), "Scratch")
    qc = build_gate_circuit(gate, Ctrl, Work1, Work2, l_t, l_rp, Scratch)

    for ctrl in [0, 1]:
        for w1, w2 in [("1001", "0100"), ("0000", "0010"), ("0000", "0000")]:
            ell_t = 2
            ell_rp = 3
            B = n + 3 - ell_rp
            init: Dict[int, int] = {}
            if ctrl:
                init[_qindex(qc, Ctrl[0])] = 1
            set_bits_lr(init, qc, Work1, w1)
            set_bits_lr(init, qc, Work2, w2)
            set_reg_int_le(init, qc, l_t, enc_len(ell_t, len_width))
            set_reg_int_le(init, qc, l_rp, enc_len(ell_rp, len_width))
            out = simulate_basis(qc, init)

            exp_l_t = enc_len(ell_t, len_width)
            if ctrl:
                exp_l_t ^= highest_position_encoded(w2, k=k, K=K, B=B, len_width=len_width)
                exp_l_t ^= highest_position_encoded(w1, k=k, K=K, B=B, len_width=len_width)
            if (
                get_reg_bits_lr(out, qc, Work1) != w1
                or get_reg_bits_lr(out, qc, Work2) != w2
                or get_reg_int_le(out, qc, l_t) != exp_l_t
                or get_reg_int_le(out, qc, l_rp) != enc_len(ell_rp, len_width)
                or not clean_reg(out, qc, Scratch)
            ):
                raise AssertionError(
                    f"len_update_lt mismatch ctrl={ctrl}, Work1={w1}, Work2={w2}: "
                    f"got l_t={get_reg_int_le(out, qc, l_t)}, expected {exp_l_t}"
                )

    # Test len_update_lrp: l_rp ^= right_len(Work1 valid) ^ right_len(Work2 valid), encoded as len-1.
    gate = eea.len_update_lrp_unary_gate(n=n, k=k, K=K, len_width=len_width)
    Ctrl = QuantumRegister(1, "Ctrl")
    Work1 = QuantumRegister(M, "Work1")
    Work2 = QuantumRegister(M, "Work2")
    l_t = QuantumRegister(len_width, "l_t")
    l_rp = QuantumRegister(len_width, "l_rp")
    Scratch = QuantumRegister(gate.num_qubits - (1 + 2 * M + 2 * len_width), "Scratch")
    qc = build_gate_circuit(gate, Ctrl, Work1, Work2, l_t, l_rp, Scratch)

    for ctrl in [0, 1]:
        for w1, w2 in [("1001", "0100"), ("0010", "0000"), ("0000", "0000")]:
            ell_t = 1
            ell_rp = 2
            A = ell_t + 2
            init = {}
            if ctrl:
                init[_qindex(qc, Ctrl[0])] = 1
            set_bits_lr(init, qc, Work1, w1)
            set_bits_lr(init, qc, Work2, w2)
            set_reg_int_le(init, qc, l_t, enc_len(ell_t, len_width))
            set_reg_int_le(init, qc, l_rp, enc_len(ell_rp, len_width))
            out = simulate_basis(qc, init)

            exp_l_rp = enc_len(ell_rp, len_width)
            if ctrl:
                exp_l_rp ^= right_length_encoded(w1, n=n, k=k, K=K, A=A, len_width=len_width)
                exp_l_rp ^= right_length_encoded(w2, n=n, k=k, K=K, A=A, len_width=len_width)
            if (
                get_reg_bits_lr(out, qc, Work1) != w1
                or get_reg_bits_lr(out, qc, Work2) != w2
                or get_reg_int_le(out, qc, l_t) != enc_len(ell_t, len_width)
                or get_reg_int_le(out, qc, l_rp) != exp_l_rp
                or not clean_reg(out, qc, Scratch)
            ):
                raise AssertionError(
                    f"len_update_lrp mismatch ctrl={ctrl}, Work1={w1}, Work2={w2}: "
                    f"got l_rp={get_reg_int_le(out, qc, l_rp)}, expected {exp_l_rp}"
                )


# End-to-end Algorithm-3 endpoint tests.
def selected_x_values(p: int, *, all_x: bool = False) -> List[int]:
    if all_x:
        return [x for x in range(1, p) if math.gcd(x, p) == 1]
    candidates = [1, 2, 3, p // 2, p // 2 + 1, p - 3, p - 2, p - 1]
    out = []
    for x in candidates:
        if 1 <= x < p and math.gcd(x, p) == 1 and x not in out:
            out.append(x)
    return out


def test_algorithm3_endpoints(primes: Sequence[int], *, mid_all_x: bool = False, verbose: bool = False) -> None:
    failures: List[str] = []
    total = 0
    for p in primes:
        n = p.bit_length()
        cfg = eea.get_n_config(n)
        fixed_T = cfg["T_max"]
        all_x = p <= 13 or mid_all_x
        for x in selected_x_values(p, all_x=all_x):
            expected = classical_expected_for_algorithm3(p, x)
            for label, T in [("exact", expected.exact_step_count), ("fixed", fixed_T)]:
                total += 1
                t0 = time.perf_counter()
                try:
                    result = run_algorithm3_endpoint_case(p=p, x=x, T=T, label=label)
                    reasons = check_endpoint_result(result)
                except Exception as exc:
                    reasons = [f"exception: {type(exc).__name__}: {exc}"]
                    result = None  # type: ignore
                if verbose:
                    dt = time.perf_counter() - t0
                    if result is None:
                        print(f"[FAIL] p={p:2d} x={x:2d} {label:5s} T={T:2d}: {reasons[0]} ({dt:.2f}s)")
                    else:
                        status = "PASS" if not reasons else "FAIL"
                        print(
                            f"[{status}] p={p:2d} x={x:2d} {label:5s} T={T:2d}/exact={result.exact_T:2d} "
                            f"t'={result.tprime}/{result.exp_tprime} Iter={result.iter_bit}/{result.exp_iter} "
                            f"inv={result.inverse}/{result.exp_inverse} ell_rp={result.l_rp_actual} "
                            f"aux={result.aux_clean} ctrl={result.ctrl_clean} ({dt:.2f}s)"
                        )
                if reasons:
                    if result is not None:
                        failures.append(
                            f"p={p}, x={x}, {label}, T={T}: "
                            + "; ".join(reasons)
                            + f"; Work1={result.work1_bits}; Work2={result.work2_bits}"
                        )
                    else:
                        failures.append(f"p={p}, x={x}, {label}, T={T}: " + "; ".join(reasons))
    if failures:
        shown = "\n".join("  - " + f for f in failures[:20])
        more = "" if len(failures) <= 20 else f"\n  ... and {len(failures) - 20} more"
        raise AssertionError(f"Algorithm-3 endpoint failures ({len(failures)}/{total}):\n{shown}{more}")


# Full Algorithm-1 wrapper tests.
def initialize_algorithm1_input(circuit, *, n: int, x: int) -> Dict[int, int]:
    init: Dict[int, int] = {}
    Work2 = _get_qreg(circuit, "Work2")
    set_reg_int_be(init, circuit, Work2, start=3, width=n, value=x)
    return init


def test_algorithm1_wrapper(primes: Sequence[int], *, verbose: bool = False) -> None:
    failures: List[str] = []
    for p in primes:
        n = p.bit_length()
        cfg = eea.get_n_config(n)
        for x in selected_x_values(p, all_x=(p <= 7)):
            t0 = time.perf_counter()
            try:
                qc = eea.build_modular_inversion_algorithm1_circuit(
                    n=n, p=p, len_width=cfg["len_width"], shift_width=cfg["shift_width"], T_max=cfg["T_max"]
                )
                assert_toffoli_network(qc, f"Algorithm-1 wrapper p={p}, x={x}")
                init = initialize_algorithm1_input(qc, n=n, x=x)
                final = simulate_basis(qc, init)

                out = _get_qreg(qc, "out")
                got_inv = get_reg_int_le(final, qc, out) % p
                exp_inv = pow(x, -1, p)
                dirty_regs = [
                    "Phase1", "Phase2", "Iter", "Sign", "Ctrl", "Work1", "l_t", "l_q", "l_s", "l_rp", "Aux"
                ]
                dirty = [name for name in dirty_regs if not clean_reg(final, qc, _get_qreg(qc, name))]
                work2 = _get_qreg(qc, "Work2")
                exp_work2_init: Dict[int, int] = {}
                set_reg_int_be(exp_work2_init, qc, work2, start=3, width=n, value=x)
                work2_ok = all(final[_qindex(qc, q)] == exp_work2_init.get(_qindex(qc, q), 0) for q in work2)
                reasons = []
                if got_inv != exp_inv:
                    reasons.append(f"output inverse got {got_inv}, expected {exp_inv}")
                if dirty:
                    reasons.append(f"unclean registers: {dirty}")
                if not work2_ok:
                    reasons.append(f"Work2 input not restored: {get_reg_bits_lr(final, qc, work2)}")
            except Exception as exc:
                reasons = [f"exception: {type(exc).__name__}: {exc}"]

            if verbose:
                status = "PASS" if not reasons else "FAIL"
                print(f"[{status}] Algorithm1 p={p} x={x} ({time.perf_counter() - t0:.2f}s)")
            if reasons:
                failures.append(f"p={p}, x={x}: " + "; ".join(reasons))
    if failures:
        raise AssertionError("Algorithm-1 wrapper failures:\n" + "\n".join("  - " + f for f in failures[:20]))


@dataclass(frozen=True)
class Table4Row:
    T: int
    work1: Optional[str]
    work2: Optional[str]
    ell_t: int
    ell_q: int
    ell_rp: int
    ell_s: int
    phase1: int
    phase2: int
    iter_bit: int
    sign: int


TABLE4_ROWS = [
    Table4Row(0,  "100100101", "000001101", 1, 0, 4, 0, 0, 0, 0, 0),
    Table4Row(1,  "100100101", "000011010", 1, 0, 4, 1, 0, 0, 0, 0),
    Table4Row(2,  "100100101", "000110100", 1, 0, 4, 2, 0, 1, 0, 0),
    Table4Row(3,  "101001011", "000011010", 1, 1, 4, 1, 0, 1, 0, 0),
    Table4Row(4,  "101001011", "000001101", 1, 2, 4, 0, 1, 0, 0, 0),
    Table4Row(5,  "101001011", "000011010", 1, 1, 4, 1, 1, 0, 0, 0),
    Table4Row(6,  "100001011", "000110101", 1, 0, 4, 2, 1, 1, 0, 1),
    Table4Row(7,  "100001011", "100011010", 1, 0, 4, 1, 1, 1, 0, 0),
    Table4Row(8,  "010001101", "100001011", 2, 0, 4, 0, 0, 0, 1, 0),
    Table4Row(12, "110001011", "010000010", 2, 0, 2, 0, 0, 0, 0, 0),
    Table4Row(16, "110100011", "000001001", 2, 1, 2, 2, 0, 1, 0, 0),
    Table4Row(24, "100010010", "110000001", 5, 0, 1, 0, 0, 0, 1, 0),
    Table4Row(32, None,        None,        6, 0, 0, 0, 0, 0, 0, 0),
    Table4Row(36, None,        None,        6, 0, 0, 4, 0, 0, 0, 0),
]


def table4_initial_bits(circuit, *, p: int = 37, x: int = 13, n: int = 6):
    cfg = eea.get_n_config(n)
    return initialize_algorithm3_input_bits(
        circuit, p=p, x=x, n=n, len_width=cfg["len_width"], shift_width=cfg["shift_width"]
    )[0]


def run_table4_prefix(T: int):
    p, x, n = 37, 13, 6
    cfg = eea.get_n_config(n)
    if T == 0:
        regs = eea.make_global_registers(n=n, len_width=cfg["len_width"], shift_width=cfg["shift_width"], T_max=0)
        qc = QuantumCircuit(*regs, name="TABLE4_INITIAL")
    else:
        qc = eea.build_full_steps_circuit(n=n, len_width=cfg["len_width"], shift_width=cfg["shift_width"], T_max=T)
        assert_toffoli_network(qc, f"Table4 prefix T={T}")
    init = table4_initial_bits(qc, p=p, x=x, n=n)
    final = simulate_basis(qc, init)
    return qc, final


def test_table4_trace(verbose: bool = False) -> None:
    failures: List[str] = []
    for row in TABLE4_ROWS:
        t0 = time.perf_counter()
        try:
            qc, final = run_table4_prefix(row.T)
            Work1 = _get_qreg(qc, "Work1")
            Work2 = _get_qreg(qc, "Work2")
            l_t = _get_qreg(qc, "l_t")
            l_q = _get_qreg(qc, "l_q")
            l_rp = _get_qreg(qc, "l_rp")
            l_s = _get_qreg(qc, "l_s")
            Phase1 = _get_qreg(qc, "Phase1")
            Phase2 = _get_qreg(qc, "Phase2")
            Iter = _get_qreg(qc, "Iter")
            Sign = _get_qreg(qc, "Sign")
            Aux = _get_qreg(qc, "Aux")
            Ctrl = _get_qreg(qc, "Ctrl")
            cfg = eea.get_n_config(6)
            got = {
                "ell_t": dec_len(get_reg_int_le(final, qc, l_t), cfg["len_width"]),
                "ell_q": dec_len(get_reg_int_le(final, qc, l_q), cfg["len_width"]),
                "ell_rp": dec_len(get_reg_int_le(final, qc, l_rp), cfg["len_width"]),
                "ell_s": dec_len(get_reg_int_le(final, qc, l_s), cfg["shift_width"]),
                "phase1": final[_qindex(qc, Phase1[0])],
                "phase2": final[_qindex(qc, Phase2[0])],
                "iter": final[_qindex(qc, Iter[0])],
                "sign": final[_qindex(qc, Sign[0])],
                "work1": get_reg_bits_lr(final, qc, Work1),
                "work2": get_reg_bits_lr(final, qc, Work2),
                "aux_clean": clean_reg(final, qc, Aux),
                "ctrl_clean": final[_qindex(qc, Ctrl[0])] == 0,
            }
            reasons = []
            for key, exp in [
                ("ell_t", row.ell_t), ("ell_q", row.ell_q), ("ell_rp", row.ell_rp), ("ell_s", row.ell_s),
                ("phase1", row.phase1), ("phase2", row.phase2), ("iter", row.iter_bit), ("sign", row.sign),
            ]:
                if got[key] != exp:
                    reasons.append(f"{key} got {got[key]}, expected {exp}")
            if row.work1 is not None and got["work1"] != row.work1:
                reasons.append(f"Work1 got {got['work1']}, expected {row.work1}")
            if row.work2 is not None and got["work2"] != row.work2:
                reasons.append(f"Work2 got {got['work2']}, expected {row.work2}")
            if not got["aux_clean"]:
                reasons.append("Aux is not clean")
            if not got["ctrl_clean"]:
                reasons.append("Ctrl is not clean")
        except Exception as exc:
            reasons = [f"exception: {type(exc).__name__}: {exc}"]
        if verbose:
            status = "PASS" if not reasons else "FAIL"
            print(f"[{status}] Table4 T={row.T:2d} ({time.perf_counter() - t0:.2f}s)")
        if reasons:
            failures.append(f"T={row.T}: " + "; ".join(reasons))
    if failures:
        raise AssertionError("PDF Table-4 trace failures:\n" + "\n".join("  - " + f for f in failures))


# Main harness.
@dataclass
class TestItem:
    name: str
    fn: callable


def main() -> None:
    parser = argparse.ArgumentParser(description="Stricter tests for eea_circuit.py against the PDF Algorithm-3 implementation")
    parser.add_argument("--primes", type=int, nargs="*", default=[3, 5, 7, 11, 13, 17],
                        help="primes used for Algorithm-3 endpoint tests; default includes n=5 p=17")
    parser.add_argument("--alg1-primes", type=int, nargs="*", default=[3, 5, 7],
                        help="small primes used for full Algorithm-1 wrapper tests")
    parser.add_argument("--mid-all-x", action="store_true",
                        help="test all x for primes above 13 as well; default tests selected edge/symmetric x")
    parser.add_argument("--table4", action="store_true", help="include heavier PDF Table-4 p=37,x=13 trace checks")
    parser.add_argument("--verbose", action="store_true", help="print each subcase")
    parser.add_argument("--skip-structure", action="store_true")
    parser.add_argument("--skip-blocks", action="store_true")
    parser.add_argument("--skip-endpoint", action="store_true")
    parser.add_argument("--skip-alg1", action="store_true")
    args = parser.parse_args()

    tests: List[TestItem] = []
    if not args.skip_structure:
        tests.extend([
            TestItem("no small-reference endpoint shortcut", test_no_small_reference_shortcut),
            TestItem("Algorithm-3 dashed-block schedule structure", test_step_schedule_structure),
            TestItem("Section-4.5 active-window formulas", test_active_window_formulas),
        ])
    if not args.skip_blocks:
        tests.extend([
            TestItem("unary iteration selector", test_unary_iteration_selector),
            TestItem("pre/post shift blocks", test_pre_post_shift_blocks),
            TestItem("phase update block", test_phase_update_block),
            TestItem("location-controlled swap block", test_lc_swap_unary_block),
            TestItem("length-update blocks", test_length_update_blocks),
        ])
    if not args.skip_endpoint:
        tests.append(TestItem(
            "Algorithm-3 endpoint exact/fixed cases",
            lambda: test_algorithm3_endpoints(args.primes, mid_all_x=args.mid_all_x, verbose=args.verbose),
        ))
    if not args.skip_alg1:
        tests.append(TestItem(
            "full Algorithm-1 wrapper primitive/functionality",
            lambda: test_algorithm1_wrapper(args.alg1_primes, verbose=args.verbose),
        ))
    if args.table4:
        tests.append(TestItem("PDF Table-4 p=37,x=13 trace", lambda: test_table4_trace(verbose=args.verbose)))

    print("Strict EEA/PDF implementation tests")
    print("These tests check more than the endpoint inverse: structure, primitive basis, blocks, endpoints, and optional Table 4 trace.\n")

    failed: List[Tuple[str, str]] = []
    t_all = time.perf_counter()
    for item in tests:
        t0 = time.perf_counter()
        try:
            item.fn()
            print(f"[PASS] {item.name} ({time.perf_counter() - t0:.2f}s)")
        except Exception as exc:
            print(f"[FAIL] {item.name} ({time.perf_counter() - t0:.2f}s)")
            print(f"       {type(exc).__name__}: {exc}")
            failed.append((item.name, f"{type(exc).__name__}: {exc}"))

    print(f"\nSummary: {len(tests) - len(failed)}/{len(tests)} test groups passed in {time.perf_counter() - t_all:.2f}s")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
