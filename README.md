# Space-Efficient Quantum Algorithm for Elliptic-Curve Discrete Logarithms with Resource Estimation

This repository contains code for resource estimation of the space-efficient quantum modular inversion procedure used in elliptic-curve discrete logarithm settings.

The project currently includes two main parts:

- `eea_model/`: a classical state-transition model of the space-efficient Extended Euclidean Algorithm (EEA).
- `eea_circuits.py`: quantum circuit construction and resource-estimation code for the modular inversion procedure.

---

## Repository structure

```text
├── eea_model/
│   ├── register.py
│   ├── one_iter.py
│   ├── one_iter_opt.py
│   └── main.py
│
├── eea_circuits.py
````

---

## Requirements

Recommended environment:

* Python 3.10+
* Qiskit

You may install the main dependency with:

```bash
pip install qiskit
```

If you would like to run the classical trace script in `eea_model/`, you may also need:

```bash
pip install pandas
```

---

## 1. `eea_model/`: classical EEA state-transition model

This folder contains a classical simulation of the quantum-friendly EEA iteration logic.

### Files

* `register.py`: register/state definitions
* `one_iter.py`: baseline single-iteration implementation
* `one_iter_opt.py`: optimized single-iteration implementation
* `main.py`: driver script for running the iteration trace and exporting intermediate states

### Run

From the repository root:

```bash
python eea_model/main.py
```

This script initializes the registers, repeatedly applies either the baseline or optimized one-iteration routine, verifies the modular inverse at the end, and writes the intermediate trace to `result.csv`.

---

## 2. `eea_circuits.py`: modular inversion circuit construction and resource estimation

The main script is:

```text
eea_circuits.py
```

This script constructs modular inversion circuit components and reports gate counts in the `ccx`, `cx`, and `x` basis. It supports two execution modes:

* **full**: explicitly builds the full circuit and transpiles it to the primitive basis
* **recursive**: counts each module recursively without constructing the full large circuit

The script contains built-in configurations for the following problem sizes:

* `n = 64, 128, 160, 192, 224, 256, 384, 512`

For these values, the corresponding `len_width` and `T_max` are predefined inside the code.

### Mode selection

The script supports:

* `--mode auto`
* `--mode full`
* `--mode recursive`

In `auto` mode, the code uses the following policy:

* `n <= 256` → `full`
* `n = 384, 512` → `recursive`

This is the default behavior implemented in the script.

---

## Running `eea_circuits.py`

### Basic usage

```bash
python eea_circuits.py --n 256
```

This runs with `mode=auto` and writes outputs to the default `outputs/` directory.

### Command-line arguments

```bash
python eea_circuits.py \
    --n 256 \
    --mode auto \
    --inspect 0 \
    --log_every 20 \
    --outdir outputs
```

Arguments:

* `--n`: problem size
* `--mode`: one of `auto`, `full`, `recursive`
* `--inspect`: print the first several step-wise module counts; mainly useful in recursive mode
* `--log_every`: logging frequency in recursive mode
* `--outdir`: output directory
* `--save_qasm`: save the full circuit QASM file; only meaningful for full mode
---

## Output files

Depending on the mode, the script produces different outputs.

### Full mode

The script prints a final summary including:

* `ccx`
* `cx`
* `x`
* total operation count
* elapsed time

If `--save_qasm` is enabled, it also saves the full circuit QASM file.

### Recursive mode

The script writes step-wise and module-wise count logs to the output directory:

* `n{n}_step_counts_recursive.txt`
* `n{n}_module_counts_recursive.txt`

For example:

```text
outputs/n512_step_counts_recursive.txt
outputs/n512_module_counts_recursive.txt
```

It also prints running progress in the terminal and a final summary at the end.
