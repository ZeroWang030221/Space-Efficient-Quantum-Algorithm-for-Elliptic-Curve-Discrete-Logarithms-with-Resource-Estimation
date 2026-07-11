import argparse
import json
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    from qiskit import QuantumCircuit
except Exception:  # pragma: no cover
    QuantumCircuit = Any  # type: ignore


PRIMITIVE_COUNTS = {
    "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "sxdg",
    "rx", "ry", "rz", "p", "u", "u1", "u2", "u3",
    "cx", "cy", "cz", "ch", "ccx", "cp", "cu", "cu1", "cu3",
    "measure", "reset", "barrier", "delay", "id",
}
CONTROL_FLOW = {"if_else", "for_loop", "while_loop", "switch_case"}


@dataclass(frozen=True)
class CounterPolicy:
    mcx_policy: str = "clean-vchain"
    expand_swap_to_cx: bool = True
    skip_alg3_steps: bool = False
    stop_prefixes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def iter_items(circ: Any):
    for item in getattr(circ, "data", []):
        if hasattr(item, "operation"):
            yield item.operation, tuple(item.qubits), tuple(item.clbits)
        else:  # qiskit < 1.0 style tuple
            yield item


def is_mcx(inst: Any) -> bool:
    name = getattr(inst, "name", "").lower()
    return name == "mcx" or name.startswith("mcx_") or name.startswith("mcx-")


def mcx_num_controls(inst: Any) -> int:
    if hasattr(inst, "num_ctrl_qubits"):
        try:
            return int(inst.num_ctrl_qubits)
        except Exception:
            pass
    return max(0, int(getattr(inst, "num_qubits", 1)) - 1)


def mcx_as_counter(inst: Any, policy: CounterPolicy) -> Counter:
    k = mcx_num_controls(inst)
    if k <= 0:
        return Counter({"x": 1})
    if k == 1:
        return Counter({"cx": 1})
    if k == 2:
        return Counter({"ccx": 1})
    if policy.mcx_policy == "clean-vchain":
        return Counter({"ccx": 2 * k - 3, "meta::mcx_expanded_clean_vchain": 1})
    if policy.mcx_policy == "keep":
        return Counter({f"mcx_{k}": 1})
    raise ValueError(f"unsupported mcx policy {policy.mcx_policy!r}")


def summarize_counter(ops: Counter, *, policy: Optional[CounterPolicy] = None) -> dict[str, Any]:
    out = {
        "x": int(ops.get("x", 0)),
        "cx": int(ops.get("cx", 0)),
        "ccx": int(ops.get("ccx", 0)),
        "h": int(ops.get("h", 0)),
        "z": int(ops.get("z", 0)),
        "measure": int(ops.get("measure", 0)),
        "reset": int(ops.get("reset", 0)),
        "barrier": int(ops.get("barrier", 0)),
        "total_counted_terms": int(sum(v for k, v in ops.items() if not k.startswith("meta::"))),
        "opaque_terms": {k: int(v) for k, v in sorted(ops.items()) if k.startswith("OPAQUE::")},
        "stopped_terms": {k: int(v) for k, v in sorted(ops.items()) if k.startswith("STOP::")},
        "meta_terms": {k: int(v) for k, v in sorted(ops.items()) if k.startswith("meta::")},
        "non_x_cx_ccx_terms": {
            k: int(v) for k, v in sorted(ops.items())
            if k not in {"x", "cx", "ccx"} and not k.startswith(("meta::", "OPAQUE::", "STOP::"))
        },
    }
    if policy is not None:
        out["policy"] = policy.as_dict()
    return out


def _definition_cache_key(inst: Any) -> tuple[Any, ...]:
    definition = getattr(inst, "definition", None)
    return (
        getattr(inst, "name", "<unnamed>"),
        int(getattr(inst, "num_qubits", 0)),
        int(getattr(inst, "num_clbits", 0)),
        id(definition),
    )


def count_instruction_recursive(inst: Any, *, policy: CounterPolicy, cache: dict[tuple[Any, ...], Counter]) -> Counter:
    name = getattr(inst, "name", "<unnamed>").lower()

    if policy.skip_alg3_steps and name.startswith("alg3_step_real_t"):
        return Counter({"meta::skipped_alg3_step": 1})

    for pref in policy.stop_prefixes:
        if name.startswith(pref.lower()):
            return Counter({f"STOP::{name}": 1})

    if is_mcx(inst):
        return mcx_as_counter(inst, policy)

    if name == "swap":
        if policy.expand_swap_to_cx:
            return Counter({"cx": 3, "meta::swap_expanded_to_3cx": 1})
        return Counter({"swap": 1})

    if name in PRIMITIVE_COUNTS:
        return Counter({name: 1})

    if name in CONTROL_FLOW and hasattr(inst, "blocks"):
        total = Counter({f"meta::control_flow_{name}": 1})
        for block in inst.blocks:
            total += count_circuit_recursive(block, policy=policy, cache=cache)
        return total

    definition = getattr(inst, "definition", None)
    if definition is None:
        return Counter({f"OPAQUE::{name}": 1})

    key = _definition_cache_key(inst)
    if key in cache:
        cached = cache[key].copy()
        cached["meta::cache_hit"] += 1
        return cached

    total = count_circuit_recursive(definition, policy=policy, cache=cache)
    cache[key] = total.copy()
    return total


def count_circuit_recursive(qc: Any, *, policy: Optional[CounterPolicy] = None, cache: Optional[dict[tuple[Any, ...], Counter]] = None) -> Counter:
    if policy is None:
        policy = CounterPolicy()
    if cache is None:
        cache = {}
    total = Counter()
    for inst, _qargs, _cargs in iter_items(qc):
        total += count_instruction_recursive(inst, policy=policy, cache=cache)
    return total


def count_gate_or_circuit(obj: Any, *, policy: Optional[CounterPolicy] = None) -> Counter:
    if policy is None:
        policy = CounterPolicy()
    if hasattr(obj, "data"):
        return count_circuit_recursive(obj, policy=policy)
    definition = getattr(obj, "definition", None)
    if definition is None:
        return Counter({f"OPAQUE::{getattr(obj, 'name', '<unnamed>').lower()}": 1})
    return count_circuit_recursive(definition, policy=policy)


def _demo() -> None:
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(5)
    qc.mcx([0, 1, 2, 3], 4)
    qc.swap(0, 4)
    policy = CounterPolicy()
    ops = count_circuit_recursive(qc, policy=policy)
    print(json.dumps(summarize_counter(ops, policy=policy), indent=2, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser(description="Small demo for the recursive CCX counter.")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not args.demo:
        ap.error("This module is normally imported by count_point_addition_ccx_by_block.py; use --demo for a self-test.")
    from io import StringIO
    # Just print to stdout; if --out is requested, write the same JSON.
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(5)
    qc.mcx([0, 1, 2, 3], 4)
    qc.swap(0, 4)
    policy = CounterPolicy()
    ops = count_circuit_recursive(qc, policy=policy)
    text = json.dumps(summarize_counter(ops, policy=policy), indent=2, sort_keys=True)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
