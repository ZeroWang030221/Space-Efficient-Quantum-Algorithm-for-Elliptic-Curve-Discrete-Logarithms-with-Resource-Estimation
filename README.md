# Quantum Algorithm for Elliptic Curve Discrete Logarithms with Space-Efficient Point Addition

This repository contains Qiskit code for resource estimation of the space-efficient quantum modular inversion and affine point-addition circuits used in elliptic-curve discrete logarithm settings.

The current codebase is centered on three workflows:

1. checkpointed recursive counting of the modular inversion circuit;
2. optional local NCT-template optimization of the circuit;
3. compiled blockwise resource estimation of the wrapped affine point-addition circuit.


---

## Repository structure

```text
.
├── README.md
│
├── eea_model/: original classical EEA reference implementation used for algorithm prototyping and correctness validation.
│
├── run_eea_s835_fastdual_recursive_chunks_checkpoint.py
├── run_eea_s835_fastdual_recursive_chunks_checkpoint_nctopt.py
├── count_s835_fastdual_wrapped_point_addition_blocks_compiled.py
│
├── eea_circuit.py
├── eea_circuit_s835_fastdual.py
├── eea_circuit_s835_lowaux.py
├── eea_circuit_updated.py
├── under1000_eea_shared_s835_fastdual_wrapped.py
├── under1000_modular_arithmetic_base.py
│
├── point_addition_fig14_s835_fastdual_wrapped_quadratic.py
├── quadratic_fig15_inplace_s835_fastdual_wrapped.py
├── quadratic_gidney_arithmetic.py
├── quadratic_lazy_instruction.py
├── quadratic_modular_arithmetic.py
├── quadratic_squ_minus.py
│
├── ccx_recursive_block_counter.py
├── nct_template_segment_optimizer.py
│
├── test_eea_strict_main.py
└── test_point_addition_strict_main.py
```

### Main entry scripts

- `run_eea_s835_fastdual_recursive_chunks_checkpoint.py`  
  Counts the EEA Algorithm-3 steps recursively, in checkpointed chunks.

- `run_eea_s835_fastdual_recursive_chunks_checkpoint_nctopt.py`  
  Same EEA counting workflow, but with bounded fail-open local NCT-template optimization.

- `count_s835_fastdual_wrapped_point_addition_blocks_compiled.py`  
  Counts the wrapped point-addition circuit by recursively counting reusable compiled subblocks and assembling the repeated arithmetic components with exact multiplicities.

### Core EEA files

- `eea_circuit_s835_fastdual.py`: production S835 fast-dual backend.  For `n=256`, the wrapped point-addition layout uses `835 = 1 + 3*256 + 66` qubits.
- `eea_circuit_s835_lowaux.py`: lower-auxiliary EEA support routines used by the fast-dual backend.
- `eea_circuit_updated.py`: shared EEA building blocks, active-window logic, Algorithm-3 step construction, and recursive operation counting helpers.
- `eea_circuit.py`: compatibility shim for `test_eea_strict_main.py`; it re-exports `eea_circuit_s835_fastdual.py` under the historical module name `eea_circuit`.

### Point-addition and arithmetic files

- `point_addition_fig14_s835_fastdual_wrapped_quadratic.py`: builds the wrapped affine point-addition circuit corresponding to the Fig.14 schedule.
- `quadratic_fig15_inplace_s835_fastdual_wrapped.py`: builds the Fig.15 in-place division and in-place multiplication structure with EEA, multiplication, measurement, reset, and feed-forward phase correction.
- `quadratic_modular_arithmetic.py`: modular addition/subtraction, multiplication, inverse multiplication, doubling, and halving instructions used by the point-addition counter.
- `quadratic_gidney_arithmetic.py`: Gidney-style arithmetic primitives and measurement/feed-forward helpers used by the quadratic modular arithmetic layer.
- `quadratic_squ_minus.py`: square-minus block used in the affine point-addition schedule.
- `under1000_eea_shared_s835_fastdual_wrapped.py`: shared EEA wrapper and S835 layout helper used by the point-addition circuit.
- `under1000_modular_arithmetic_base.py`: small shared modular-arithmetic utilities.

### Counting and optimization utilities

- `ccx_recursive_block_counter.py`: recursive counter for Qiskit circuits/instructions, with policies for MCX expansion and SWAP expansion.
- `nct_template_segment_optimizer.py`: local template/cancellation optimizer for reversible `{X, CX, CCX}` segments.

---

## Requirements

Recommended environment:

- Python 3.10+
- Qiskit

Install the main dependency with:

```bash
python -m pip install --upgrade pip
python -m pip install qiskit
```

---

## Quick start

Run the test suite:

```bash
python test_eea_strict_main.py
python test_point_addition_strict_main.py
```

For a faster point-addition smoke test:

```bash
python test_point_addition_strict_main.py --skip-n256 --skip-report
```

---

## 1. EEA Algorithm-3 recursive chunk counting

The standard EEA counting entry point is:

```bash
python run_eea_s835_fastdual_recursive_chunks_checkpoint.py \
  --n 192 \
  --chunk-size 25 \
  --measurement-uncompute \
  --resume \
  --workdir eea_s835_fastdual_chunks25 \
  --out eea_s835_fastdual_algorithm3_recursive_chunks_n192_measurement.json
```

Important arguments:

- `--n`: bit width.
- `--T-max`: optional override for the number of Algorithm-3 steps; by default the value from `eea.get_n_config(n)` is used.
- `--chunk-size`: number of Algorithm-3 steps counted per checkpoint chunk.
- `--aux-size`: optional override for the helper-qubit pool; if omitted, the paper/S835 layout helper size is computed automatically.
- `--measurement-uncompute`: enables measurement-based uncomputation in the counted EEA blocks.
- `--resume`: reuses existing non-empty chunk JSON files in `--workdir`.
- `--workdir`: directory for per-chunk checkpoint files.
- `--out`: cumulative JSON summary written after every chunk.

The script writes per-chunk files such as:

```text
eea_s835_fastdual_chunks25/eea_s835_fastdual_n192_T0001_0025.json
```

and a cumulative output JSON containing fields such as:

```text
mode
n
T_max
num_qubits
len_width
shift_width
aux_size
measurement_based
ops
chunks
elapsed_s_so_far
```

---

## 2. EEA counting with bounded NCT-template optimization

The optimized counting entry point is:

```bash
python run_eea_s835_fastdual_recursive_chunks_checkpoint_nctopt.py \
  --n 128 \
  --chunk-size 25 \
  --measurement-uncompute \
  --templates small-nct \
  --rounds 1 \
  --max-nct-segment-gates 40 \
  --segment-timeout-s 10 \
  --timeout-mode auto \
  --resume \
  --workdir eea_s835_fastdual_chunks_nctopt_failopen_r1_128_seg40_to10 \
  --out eea_s835_fastdual_algorithm3_recursive_chunks_n128_measurement_nctopt_failopen_r1_seg40_to10.json
```

This workflow attempts local template optimization on reversible `{X, CX, CCX}` segments.  It is designed as a bounded fail-open counter: if an optimized step times out or raises an exception, that step is counted exactly without template rounds and then checkpointed, so the final reported counts remain complete.

Useful arguments in addition to the standard EEA arguments:

- `--templates {small-nct,all-nct}`: template library selection.
- `--rounds`: number of template-optimization rounds.
- `--max-nct-segment-gates`: maximum size of a reversible segment sent to template optimization.
- `--max-nct-segment-qubits`: maximum number of qubits in a segment.
- `--segment-timeout-s`: timeout for individual segment optimization.
- `--step-timeout-s`: timeout for a whole Algorithm-3 step before falling back to unchanged counting.
- `--fallback-step-timeout-s`: timeout for the exact fallback count.
- `--force`: recompute even if step/chunk checkpoints already exist.
- `--ignore-policy-mismatch`: reuse old checkpoints even when the optimization policy differs; this is mainly for debugging.

The optimized workflow writes both step-level checkpoints under:

```text
<workdir>/steps/
```

and chunk-level summaries under:

```text
<workdir>/
```

---

## 3. Wrapped S835 point-addition compiled blockwise counting

The point-addition counter depends on an EEA Algorithm-3 JSON produced by one of the EEA workflows above.  The `--n` value of the point-addition counter should match the `n` field in the EEA JSON.

Example for `n=64`:

```bash
python run_eea_s835_fastdual_recursive_chunks_checkpoint.py \
  --n 64 \
  --chunk-size 25 \
  --measurement-uncompute \
  --resume \
  --workdir eea_s835_fastdual_chunks25_n64 \
  --out eea_s835_fastdual_algorithm3_recursive_chunks_n64_measurement.json
```

Then run:

```bash
python count_s835_fastdual_wrapped_point_addition_blocks_compiled.py \
  --n 64 \
  --eea-steps-json eea_s835_fastdual_algorithm3_recursive_chunks_n64_measurement.json \
  --out point_addition_s835_fastdual_wrapped_blocks_compiled_counts_n64.json
```

Example for the optimized `n=128` EEA output:

```bash
python count_s835_fastdual_wrapped_point_addition_blocks_compiled.py \
  --n 128 \
  --eea-steps-json eea_s835_fastdual_algorithm3_recursive_chunks_n128_measurement_nctopt_failopen_r1_seg40_to10.json \
  --out point_addition_s835_fastdual_wrapped_blocks_compiled_counts_n128.json
```

Important arguments:

- `--n`: bit width.
- `--p`: modulus; defaults to the secp256k1 prime.
- `--s-qubits`: optional override for the shared EEA arithmetic register size.
- `--point-constant {secp256k1-generator,zero,custom}`: point constant selection for the Fig.14 constant-coordinate updates.
- `--x2`, `--y2`: custom point coordinates; required when `--point-constant custom` is used.
- `--eea-steps-json`: JSON file containing recursive Algorithm-3 EEA counts.
- `--allow-eea-n-mismatch`: debug-only override allowing the EEA JSON `n` to differ from the requested `--n`.
- `--mcx-policy {clean-vchain,keep}`: MCX expansion policy for recursive counting.
- `--validate-full-mul`: for small `n`, recursively count full multiplication/squaring definitions and compare them with the assembled block counts.
- `--out`: output JSON path.

The output report includes:

```text
counting_mode
n
p
point_constant_kind
qiskit_width_report
eea_meta
block_summaries
raw_block_counters
key_ccx
validation
elapsed_s
```

The point-addition counter is not a closed-form formula.  It builds reusable Qiskit circuits/instructions, recursively counts them in the `{CCX, CX, X}` basis, and then assembles larger repeated blocks such as multiplication, inverse multiplication, in-place division, in-place multiplication, square-minus, and the total Fig.14 point-addition block.

---

## Tests

This repository includes two plain Python test drivers.  They are intentionally written without `pytest`, Aer, or full statevector simulation.  The tests recursively expand Qiskit definitions where appropriate and simulate computational-basis states for Toffoli-network blocks.

### EEA strict tests

The EEA tests are in:

```text
test_eea_strict_main.py
```

Run the default EEA suite with:

```bash
python test_eea_strict_main.py
```

The default suite checks:

- the implementation is not a small-`n` endpoint shortcut;
- the Algorithm-3 dashed-block schedule is present;
- active-window formulas over several small widths;
- unary iteration, pre/post shift, phase update, location-controlled swap, and length-update blocks;
- Algorithm-3 endpoints for the default primes `3, 5, 7, 11, 13, 17`, using both exact step counts and fixed `T_max`;
- the full Algorithm-1 wrapper for small default primes `3, 5, 7`.

Useful variants:

```bash
# Fast structural + block tests only.
python test_eea_strict_main.py --skip-endpoint --skip-alg1

# Include the heavier PDF/Table-4 p=37, x=13 trace benchmark.
python test_eea_strict_main.py --table4

# Test all x values for primes above 13 as well.
python test_eea_strict_main.py --primes 3 5 7 11 13 17 --mid-all-x --verbose
```

### Point-addition strict tests

The point-addition tests are in:

```text
test_point_addition_strict_main.py
```

Run the default point-addition suite with:

```bash
python test_point_addition_strict_main.py
```

The default point-addition suite checks:

- S835 wrapped point-addition register layout;
- the `n=256` width identity `835 = 1 + 3*256 + 66`;
- Fig.14/Fig.15 top-level operation order;
- explicit dynamic-circuit structure involving `H`, `measure`, `reset`, classically controlled `Z`, and `swap` operations;
- compiled arithmetic subblock assembly for a small modulus;
- a tiny integrated point-addition counter report using a synthetic Algorithm-3 JSON, so the report path is tested without running a large EEA count.

The large-prime regression matrix covers representative pairs of the field bit width `n` and prime modulus `p`, ranging from 12-bit to 512-bit prime fields.  The tested instances include, for example, `n=16, p=65521`, `n=32, p=4294967291`, the secp256k1 prime at `n=256`, and representative primes at `n=128, 160, 192, 224, 384, 512`.

For each `(n,p)` pair, the tests include boundary, symmetric, random, and relatively long EEA traces.  Full compiled-arithmetic assembly is run only for selected moderate-width instances, while the larger `(n,p)` pairs are used to validate circuit construction, register layout, scheduling, and recursive resource-counting paths.

Useful variants:

```bash
# Skip the tiny integrated report and only check construction/schedule/assembly.
python test_point_addition_strict_main.py --skip-report

# Fast smoke test that also skips the n=256 width construction check.
python test_point_addition_strict_main.py --skip-n256 --skip-report

# Use a different small prime/width for compiled-block validation.
python test_point_addition_strict_main.py --n 5 --p 17
```

If Qiskit is not installed, `test_point_addition_strict_main.py` prints a skip message and exits successfully.  The EEA strict test requires Qiskit because it builds the EEA/PDF block gates.

---

## Reproducing the main statistics

Our paper reports numerical resource-estimation results for:

```text
n = 64, 128, 160, 192, 224, 256, 384, 512
```

A typical workflow is:

1. run the EEA Algorithm-3 counter for a given `n`;
2. optionally run the NCT-optimized version for the same `n`;
3. pass the resulting EEA JSON to the wrapped point-addition counter;
4. collect the `key_ccx`, `block_summaries`, and `qiskit_width_report` fields from the output report.

For large widths, use `--resume` and keep the `--workdir` directories, since chunk and step checkpoints are meant to support interrupted long runs.
