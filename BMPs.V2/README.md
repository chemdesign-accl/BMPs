# BMPs V2

BMPs V2 is a molecular graph neural network framework for molecular property
prediction and atom-level interpretation. It builds on the original BMPs V1
architecture used in Digital Discovery, 2025, 4, 3320-3338, extending it with
faster reusable preprocessing, improved stereoisomer/conformer handling,
configurable normalization, message-passing depth comparison, and revised
node-level importance scores for molecular interpretation.

The installable package name is `BMPs`, and the Python import name is `BMPs`.

The model uses atom, bond, and global molecular descriptors extracted from RDKit
and supports both classification and regression workflows.

If you use BMPs, please cite: Digital Discovery, 2025, 4, 3320-3338.

## Main Updates From V1

- Processed-molecule caching for reuse across repeated runs and
  cross-validation.
- Multi-CPU preprocessing for molecule graph generation.
- Revised stereoisomer/conformer protocol using stereoisomer enumeration,
  multiple ETKDGv3 conformers, MMFF optimization, and lowest-energy conformer
  selection.
- Safer heavy-atom graph indexing for molecules where explicit hydrogens can
  remain in RDKit atom indices.
- Optional unique-occupancy buried-volume mode for non-redundant local
  occlusion fractions.
- `graphnorm`, `layernorm`, and `none` normalization options in addition to the
  original BatchNorm-style path.
- Configurable message-passing depth with optional automatic comparison.
- Classification class imbalance handled through
  `BCEWithLogitsLoss(pos_weight=negatives/positives)` instead of DataLoader
  sampling.
- Node-level importance changed from learned 1D node scores with simple
  molecule-level min-max scaling to raw message-derived scores with `log1p` and
  robust min-max scaling.
- ABMP/ABMP+SN node-importance scores are coupled to the same edge-attention
  weights used in message passing.

## Node Blocks

| Node block | Description |
| --- | --- |
| `BMP` | Base bidirectional message passing without self-node features. |
| `ABMP` | Attention-enhanced bidirectional message passing. |
| `CBMP` | Convolutional-normalization bidirectional message passing. |
| `BMP+SN` | BMP with self-node connections. |
| `ABMP+SN` | ABMP with self-node connections. |
| `UMP` | Updated message-passing block variant. |

## Installation

The tested V2 environment uses Python 3.11, CUDA Toolkit 12.9, PyTorch 2.8.0,
and PyTorch Geometric wheels for CUDA 12.9. Start from a clean conda
environment:

```bash
conda create -n bmps_v2 python=3.11 -y
conda activate bmps_v2

conda install nvidia::cuda-toolkit==12.9.0

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu129

pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.8.0+cu129.html
```

Install V2 directly from GitHub:

```bash
pip install "git+https://github.com/chemdesign-accl/BMPs.git#subdirectory=BMPs.V2"
```

For development, clone the repository and install V2 in editable mode:

```bash
git clone https://github.com/chemdesign-accl/BMPs.git
cd BMPs/BMPs.V2
pip install -e .
```

Check that the package imports correctly:

```bash
python -c "from BMPs import GNNTrainer; print('BMPs V2 installed')"
```

## Package Layout

```text
BMPs.V2/
  setup.py
  BMPs/
    data/
      molecular_dataset.py
    model/
      blocks.py
      interaction_network.py
      node_blocks.py
      trainer.py
    examples/
      evaluate_internal_split.py
      cross_validate.py
      predict.py
```

## Data Format

Input CSV files should include:

- `SMILES`: molecular SMILES string.
- `Title`: compound or molecule name.
- target column: configurable label or continuous regression target.
- optional `Model`: split assignment column for workflows that use predefined
  train/evaluation assignments.

If no split column is available, the evaluation utilities can create random
splits, using stratification for classification when the labels allow it.

## Example Utilities

`evaluate_internal_split.py` trains a model, evaluates it on an internal split,
saves metrics, and can produce molecule images with node-level importance
values.

`cross_validate.py` performs k-fold model evaluation with cache reuse and
task-aware fold handling.

`predict.py` applies a trained model to molecules and writes prediction tables.
For classification it can report logits, probabilities, thresholds, and labels.
For regression it can report raw and inverse-transformed predictions.

Common configuration options in the example scripts include:

| Option | Purpose |
| --- | --- |
| `task` | `"Classification"` or `"Regression"`. |
| `target_column` | CSV column used as the label or regression target. |
| `node_block` | One of the supported node blocks. |
| `normalization` | `graphnorm`, `batchnorm`, `layernorm`, or `none`. |
| `message_passing_steps` | Fixed number of message-passing iterations. |
| `auto_message_passing` | Compare message-passing depths automatically. |
| `preprocessing_num_workers` | Number of CPU workers for molecule processing. |
| `preprocessing_cache_dir` | Directory for reusable processed molecules. |

## Processed-Molecule Cache

The processed cache stores graph features keyed by canonical SMILES and
preprocessing settings such as node block, conformer count, stereoisomer limit,
and buried-volume parameters. Labels are reattached when a cached molecule is
loaded, so the same feature cache can be reused across classification,
regression, cross-validation, and prediction when the molecular structures and
preprocessing settings match.

Clear or separate the cache when changing feature definitions, node features,
global features, buried-volume settings, conformer settings, or major
preprocessing logic.

## Atom Importance

Atom importance is computed from raw message-derived node scores. Scores are
processed with:

1. non-negative clipping of raw message scores.
2. `log1p` compression.
3. robust min-max scaling using quantiles.
4. high-score and low-score masks.

In produced molecule images this gives smoother, more interpretable node-level
color maps. Strongly important atoms should still stand out, while neighboring
or moderately relevant atoms remain visible.

The high-score atoms can be treated as conserved core/pharmacophoric regions,
while lower-scoring atoms can be prioritized for bioisostere-replacement search.
Node-score text files include raw message score, transformed score, normalized
score, high-score flag, and low-score flag.
