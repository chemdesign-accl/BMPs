import logging
import os
import time

import numpy as np
import pandas as pd
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader as GeometricDataLoader

from BMPs import GNNTrainer
from BMPs.data.molecular_dataset import MolecularDataset


logging.basicConfig(
    filename='predict.log',
    filemode='w',
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# Config
task = "Classification"
target_column = "Actividad"
training_filename = "TRPA1_for_training.csv"
prediction_filename = "bace.csv"
node_block = "ABMP"
normalization = "graphnorm"
hidden_channels = 217
dropout_rate = 0.26
lr = 1e-3
batch_size = 155
epochs = 150
threshold = 0.5
standardize_regression_target = True
message_passing_steps = 1
preprocessing_num_workers = min(128, os.cpu_count() or 1)

script_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(script_dir, "predict_outputs")
os.makedirs(input_dir, exist_ok=True)
preprocessing_cache_dir = os.path.join(input_dir, "processed_molecule_cache")
output_csv_path = os.path.join(input_dir, "predictions.csv")


def require_columns(frame, columns, dataset_name):
    missing_columns = set(columns).difference(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {dataset_name}: {sorted(missing_columns)}"
        )


def get_names(frame):
    if "Title" in frame.columns:
        return frame["Title"].astype(str).tolist()
    return [f"molecule_{index + 1}" for index in range(len(frame))]


start = time.time()
if task not in {"Classification", "Regression"}:
    raise ValueError("task must be either 'Classification' or 'Regression'.")

training_path = os.path.join(script_dir, "..", "data", training_filename)
prediction_path = os.path.join(script_dir, "..", "data", prediction_filename)
train_df = pd.read_csv(training_path)
predict_df = pd.read_csv(prediction_path, dtype={0: str})

require_columns(train_df, {"SMILES", target_column}, training_path)
require_columns(predict_df, {"SMILES"}, prediction_path)

smiles_train = train_df["SMILES"].tolist()
labels_train = train_df[target_column].tolist()
names_train = get_names(train_df)
smiles_predict = predict_df["SMILES"].tolist()
names_predict = get_names(predict_df)

regression_target_mean = 0.0
regression_target_std = 1.0
if task == "Regression":
    labels_train_array = np.asarray(labels_train, dtype=float)
    regression_target_mean = float(labels_train_array.mean())
    regression_target_std = float(labels_train_array.std())
    if regression_target_std <= 0:
        logger.warning("Regression target standard deviation is zero; using unscaled targets.")
        regression_target_std = 1.0
    if standardize_regression_target:
        labels_train = [
            (float(label) - regression_target_mean) / regression_target_std
            for label in labels_train
        ]
        logger.info(
            f"Regression target standardization enabled: "
            f"mean={regression_target_mean:.6f}, std={regression_target_std:.6f}"
        )

logger.info(
    f"Prediction run configuration: task={task}, target_column={target_column}, "
    f"training_data={training_path}, prediction_data={prediction_path}, "
    f"node_block={node_block}, normalization={normalization}, "
    f"hidden_channels={hidden_channels}, dropout_rate={dropout_rate}, lr={lr}, "
    f"batch_size={batch_size}, epochs={epochs}, threshold={threshold}, "
    f"standardize_regression_target={standardize_regression_target}, "
    f"message_passing_steps={message_passing_steps}, "
    f"preprocessing_cache_dir={preprocessing_cache_dir}"
)

train_dataset = MolecularDataset(
    smiles_train,
    names_train,
    labels_train,
    node_block=node_block,
    num_workers=preprocessing_num_workers,
    cache_dir=preprocessing_cache_dir,
)

global_dim = train_dataset.global_dim
edge_dim = train_dataset.edge_dim
num_node_features = train_dataset.num_node_features
successful_labels_train = train_dataset.successful_labels
successful_smiles_train = train_dataset.successful_smiles
successful_names_train = train_dataset.successful_names
if len(successful_labels_train) == 0:
    raise ValueError("No successful training labels after preprocessing.")

if task == "Classification":
    label_array = np.asarray(successful_labels_train, dtype=int)
    label_counts = np.bincount(label_array, minlength=2)
    logger.info(
        f"Training labels after preprocessing: negatives={int(label_counts[0])}, "
        f"positives={int(label_counts[1])}"
    )
    if np.all(label_counts > 0):
        class_weights = 1.0 / label_counts
        sample_weights = class_weights[label_array]
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = GeometricDataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
        )
    else:
        logger.warning("Classification training set has one class; using shuffled loader.")
        train_loader = GeometricDataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
else:
    train_targets = np.asarray(successful_labels_train, dtype=float)
    if standardize_regression_target:
        train_targets = train_targets * regression_target_std + regression_target_mean
    logger.info(
        f"Training target after preprocessing: count={len(train_targets)}, "
        f"mean={train_targets.mean():.6f}, std={train_targets.std():.6f}, "
        f"min={train_targets.min():.6f}, max={train_targets.max():.6f}"
    )
    train_loader = GeometricDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

trainer = GNNTrainer(
    smiles_list=successful_smiles_train,
    labels=successful_labels_train,
    names_list=successful_names_train,
    hidden_channels=hidden_channels,
    num_node_features=num_node_features,
    global_dim=global_dim,
    lr=lr,
    edge_dim=edge_dim,
    batch_size=batch_size,
    dropout_rate=dropout_rate,
    input_dir=input_dir,
    node_block=node_block,
    task=task,
    threshold=threshold,
    normalization=normalization,
    preprocessing_num_workers=preprocessing_num_workers,
    preprocessing_cache_dir=preprocessing_cache_dir,
    standardize_regression_target=standardize_regression_target,
    message_passing_steps=message_passing_steps,
)
trainer.setup_model()

logger.info(f"Starting training for {epochs} epochs.")
for epoch in range(epochs):
    epoch_start = time.time()
    loss = trainer.train(train_loader)
    epoch_time = time.time() - epoch_start
    logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Time: {epoch_time:.2f}s")
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Time: {epoch_time:.2f}s")

results = trainer.predict(
    smiles_predict,
    names_predict,
    output_csv=output_csv_path,
    regression_target_mean=regression_target_mean if standardize_regression_target else 0.0,
    regression_target_std=regression_target_std if standardize_regression_target else 1.0,
)

elapsed = time.time() - start
logger.info(f"Wrote {len(results)} predictions to {output_csv_path}. Elapsed time: {elapsed:.2f}s")
print(f"Wrote {len(results)} predictions to {output_csv_path}")
print(f"Elapsed time: {elapsed:.2f} seconds")
