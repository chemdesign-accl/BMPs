import pandas as pd
import logging
import os
from BMPs import GNNTrainer

logging.basicConfig(
    filename='cross_validate.log',
    filemode='w',
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Config
task = "Classification"
target_column = "Actividad"
dataset_filename = "bbbp.csv"
node_block = "ABMP"
normalization = "graphnorm"
hidden_channels = 64
dropout_rate = 0.2
lr = 1e-3
batch_size = 32
k_folds = 5
epochs = 200
standardize_regression_target = True
message_passing_steps = 1
auto_message_passing = False
max_message_passing_steps = 4
message_passing_min_delta = 1e-4
message_passing_patience = 1
preprocessing_num_workers = min(128, os.cpu_count() or 1)

script_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(script_dir, "cross_validate_output")
os.makedirs(input_dir, exist_ok=True)
preprocessing_cache_dir = os.path.join(input_dir, "processed_molecule_cache")

data_path = os.path.join(script_dir, "..", "data", dataset_filename)
data = pd.read_csv(data_path)
required_columns = {"SMILES", target_column, "Title"}
missing_columns = required_columns.difference(data.columns)
if missing_columns:
    raise ValueError(f"Missing required columns in dataset: {sorted(missing_columns)}")

if task not in {"Classification", "Regression"}:
    raise ValueError("task must be either 'Classification' or 'Regression'.")

smiles_train = data['SMILES'].tolist()
labels_train = data[target_column].tolist()
names_train = data['Title'].tolist()

logger.info(
    f"Cross-validation configuration: task={task}, target_column={target_column}, "
    f"dataset={data_path}, node_block={node_block}, normalization={normalization}, "
    f"hidden_channels={hidden_channels}, dropout_rate={dropout_rate}, lr={lr}, "
    f"batch_size={batch_size}, k_folds={k_folds}, epochs={epochs}, "
    f"standardize_regression_target={standardize_regression_target}, "
    f"message_passing_steps={message_passing_steps}, "
    f"auto_message_passing={auto_message_passing}, "
    f"max_message_passing_steps={max_message_passing_steps}, "
    f"preprocessing_cache_dir={preprocessing_cache_dir}"
)
if task == "Classification":
    logger.info(
        f"Label counts: negatives={int((data[target_column] == 0).sum())}, "
        f"positives={int((data[target_column] == 1).sum())}"
    )
else:
    logger.info(
        f"Target summary: count={len(data)}, mean={data[target_column].mean():.6f}, "
        f"std={data[target_column].std():.6f}, min={data[target_column].min():.6f}, "
        f"max={data[target_column].max():.6f}"
    )

trainer = GNNTrainer(
    smiles_list=smiles_train,
    labels=labels_train,
    names_list=names_train,
    hidden_channels=hidden_channels,
    lr=lr,
    batch_size=batch_size,
    k_folds=k_folds,
    dropout_rate=dropout_rate,
    input_dir=input_dir,
    epochs=epochs,
    task=task,
    node_block=node_block,
    normalization=normalization,
    preprocessing_num_workers=preprocessing_num_workers,
    preprocessing_cache_dir=preprocessing_cache_dir,
    standardize_regression_target=standardize_regression_target,
    message_passing_steps=message_passing_steps,
    auto_message_passing=auto_message_passing,
    max_message_passing_steps=max_message_passing_steps,
    message_passing_min_delta=message_passing_min_delta,
    message_passing_patience=message_passing_patience,
)
trainer.cross_validate()
