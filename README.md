# BMPs

This repository contains two installable versions of the bidirectional
message-passing molecular graph neural network codebase.

## Repository Layout

```text
BMPs/
  README.md
  BMPs.V1/
    README.md
    setup.py
    BMPNNs/
  BMPs.V2/
    README.md
    setup.py
    BMPs/
```

## Which Version Should I Use?

Use `BMPs.V1` if you need the original package associated with the Digital
Discovery work. It installs the Python package `bmpnns` and is imported as
`BMPNNs`.

Use `BMPs.V2` if you need the updated package with reusable preprocessing,
stereoisomer/conformer updates, configurable normalization, message-passing
depth comparison, and revised atom-importance scoring. It installs the Python
package `BMPs` and is imported as `BMPs`.

Install one version per Python environment when possible. This keeps PyTorch,
PyTorch Geometric, and local editable installs easy to reason about.

## Install V1

Create and activate a clean environment:

```bash
conda create -n bmps_v1 python=3.11 -y
conda activate bmps_v1
```

Install PyTorch and PyTorch Geometric dependencies following the version and
CUDA build appropriate for your machine. The V1 README contains the tested
setup notes.

Install directly from this repository:

```bash
pip install "git+https://github.com/chemdesign-accl/BMPs.git#subdirectory=BMPs.V1"
```

Or install from a local clone in editable mode:

```bash
git clone https://github.com/chemdesign-accl/BMPs.git
cd BMPs/BMPs.V1
pip install -e .
```

Check the installation:

```bash
python -c "from BMPNNs import GNNTrainer; print('BMPs V1 installed')"
```

## Install V2

Create and activate a clean environment:

```bash
conda create -n bmps_v2 python=3.11 -y
conda activate bmps_v2
```

Install PyTorch and PyTorch Geometric dependencies following the version and
CUDA build appropriate for your machine. The V2 README contains the tested
setup notes.

Install directly from this repository:

```bash
pip install "git+https://github.com/chemdesign-accl/BMPs.git#subdirectory=BMPs.V2"
```

Or install from a local clone in editable mode:

```bash
git clone https://github.com/chemdesign-accl/BMPs.git
cd BMPs/BMPs.V2
pip install -e .
```

Check the installation:

```bash
python -c "from BMPs import GNNTrainer; print('BMPs V2 installed')"
```

## Version-Specific Documentation

Each version keeps its own README, setup file, examples, data, and package
source:

- `BMPs.V1/README.md`: original V1 package notes and installation details.
- `BMPs.V2/README.md`: V2 package notes, update summary, examples, and
  installation details.

## Citation

If you use this code, please cite the associated Digital Discovery publication:
Digital Discovery, 2025, 4, 3320-3338.
