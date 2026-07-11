"""Lazy-defined Qiskit Instruction with optional classical bits.

Unlike Gate, Instruction can carry clbits, so it can be used for measurement-
based Gidney adders/comparators.  The instruction is not opaque: accessing
.definition builds the real subcircuit.
"""
from typing import Callable, Optional
from qiskit.circuit import Instruction, QuantumCircuit


class LazyDefinedInstruction(Instruction):
    def __init__(self, name: str, num_qubits: int, num_clbits: int, builder: Callable[[], QuantumCircuit], *, label: Optional[str] = None):
        super().__init__(name=name, num_qubits=int(num_qubits), num_clbits=int(num_clbits), params=[])
        self._lazy_builder = builder
        self._lazy_definition = None
        self.label = label

    @property
    def definition(self):  # type: ignore[override]
        if self._lazy_definition is None:
            self._lazy_definition = self._lazy_builder()
        return self._lazy_definition

    @definition.setter
    def definition(self, value):  # type: ignore[override]
        self._lazy_definition = value
