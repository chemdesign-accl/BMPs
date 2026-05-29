# BMPs V1

BMPs V1 is the original bidirectional message-passing neural network package
for molecular property prediction. It is distributed in this repository under
`BMPs.V1/` so it can live alongside the updated V2 package without replacing
it.

The installable package name is `bmpnns`, and the Python import name is
`BMPNNs`.

## Node Blocks

| Node model | Description |
| --- | --- |
| `BMP` | Base message passing without self-nodes. |
| `ABMP` | Attention-enhanced message passing. |
| `CBMP` | Convolutional-normalization message passing. |
| `BMP+SN` | BMP with self-nodes. |
| `ABMP+SN` | ABMP with self-nodes. |

## Installation

PyTorch Geometric and its extension packages must match your PyTorch and CUDA
versions. The tested V1 environment uses Python 3.11, GPU execution with CUDA
12.8, and PyTorch 2.8.

Create and activate a clean conda environment:

```bash
conda create -n bmps_v1 python=3.11 -y
conda activate bmps_v1
```

Install PyTorch:

```bash
pip install torch torchvision torchaudio
```

Install PyTorch Geometric and optional compiled extensions:

```bash
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
```

Install V1 directly from GitHub:

```bash
pip install "git+https://github.com/chemdesign-accl/BMPs.git#subdirectory=BMPs.V1"
```

For development, clone the repository and install V1 in editable mode:

```bash
git clone https://github.com/chemdesign-accl/BMPs.git
cd BMPs/BMPs.V1
pip install -e .
```

Check that the package imports correctly:

```bash
python -c "from BMPNNs import GNNTrainer; print('BMPs V1 installed')"
```

## Package Layout

```text
BMPs.V1/
  setup.py
  BMPNNs/
    data/
    examples/
    model/
```

## Notes

Use a separate environment for V1 when comparing against V2. This avoids
confusing editable installs and makes dependency changes easier to isolate.
