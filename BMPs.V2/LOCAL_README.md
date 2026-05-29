# BMPs

BMPs is a modular graph neural network framework for molecular property
prediction. This V2 update builds on the original BMPs implementation used in
Digital Discovery, 2025, 4, 3320-3338, and adds reusable preprocessing,
multi-CPU molecule processing, updated stereoisomer/conformer handling,
normalization options, message-passing depth comparison, and revised atom-level
importance maps.

If you use BMPs, please cite: Digital Discovery, 2025, 4, 3320-3338.

## Installation

The tested local environment uses Python 3.11, CUDA Toolkit 12.9, PyTorch 2.8.0,
and PyTorch Geometric wheels for CUDA 12.9.

```bash
conda create -n pyg_env python=3.11 -y
conda activate pyg_env

conda install nvidia::cuda-toolkit==12.9.0

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu129

pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.8.0+cu129.html
```

For regular use, install BMPs directly from GitHub:

```bash
pip install git+https://github.com/chemdesign-accl/BMPs.git@v2.0.0
```

To install V1, the version associated with the Digital Discovery paper, use the
V1 tag instead:

```bash
pip install git+https://github.com/chemdesign-accl/BMPs.git@V1
```

For development, clone the repository and install it in editable mode:

```bash
git clone https://github.com/chemdesign-accl/BMPs.git
cd BMPs
pip install -e .
```

Check the local install:

```bash
python -c "from BMPs import GNNTrainer; print('BMPs installed')"
```

## Node Models

| Node block | Description |
| --- | --- |
| `BMP` | Base bidirectional message passing without self-node features. |
| `ABMP` | Attention-enhanced bidirectional message passing. |
| `CBMP` | Convolutional-normalization bidirectional message passing. |
| `BMP+SN` | BMP with self-node connections. |
| `ABMP+SN` | ABMP with self-node connections. |
| `UMP` | Updated message-passing block variant. |

## Included Data

The repository includes MoleculeNet datasets such as BACE and BBBP, and
BindingDB-derived TRPA1 datasets used for example workflows.

## References

[1] Battaglia, P. W., Hamrick, J. B., Bapst, V., Sanchez-Gonzalez, A.,
Zambaldi, V., Malinowski, M., Tacchetti, A., Raposo, D., Santoro, A.,
Faulkner, R., Gilmer, J., Dahl, G., Vaswani, A., Allen, K., Nash, C.,
Langston, V., Dyer, C., Heess, N., Wierstra, D., Kohli, P., Botvinick, M.,
Vinyals, O., Li, Y., & Pascanu, R. (2018). Relational inductive biases, deep
learning, and graph networks. arXiv preprint arXiv:1806.01261.

[2] Wu, Z., Ramsundar, B., Feinberg, E. N., Gomes, J., Geniesse, C., Pappu,
A. S., Leswing, K., & Pande, V. (2018). MoleculeNet: a benchmark for molecular
machine learning. Chemical Science, 9(2), 513-530.

[3] Gilson, M. K., Liu, T., Baitaluk, M., Nicola, G., Hwang, L., & Chong, J.
(2016). BindingDB in 2015: a public database for medicinal chemistry,
computational chemistry and systems pharmacology. Nucleic Acids Research,
44(D1), D1045-D1053.
