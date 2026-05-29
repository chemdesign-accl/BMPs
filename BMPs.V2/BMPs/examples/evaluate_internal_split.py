
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    confusion_matrix,
    ConfusionMatrixDisplay,
    r2_score,
    roc_curve,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader as GeometricDataLoader
from BMPs import GNNTrainer
from BMPs.data.molecular_dataset import MolecularDataset
import logging

logging.basicConfig(
    filename='evaluate_internal_split.log',
    filemode='w',
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


def inverse_regression_values(values):
    values = np.asarray(values, dtype=float)
    if task == "Regression" and standardize_regression_target:
        return values * regression_target_std + regression_target_mean
    return values


def compute_regression_metrics(targets, predictions):
    targets = np.asarray(targets, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    errors = predictions - targets
    if len(targets) > 1 and np.std(targets) > 0:
        r2 = r2_score(targets, predictions)
    else:
        r2 = float("nan")
    if len(targets) > 1 and np.std(targets) > 0 and np.std(predictions) > 0:
        pearson = float(np.corrcoef(targets, predictions)[0, 1])
        spearman = float(
            pd.Series(targets).rank().corr(pd.Series(predictions).rank())
        )
    else:
        pearson = float("nan")
        spearman = float("nan")
    return {
        "rmse": root_mean_squared_error(targets, predictions),
        "mae": mean_absolute_error(targets, predictions),
        "r2": r2,
        "pearson": pearson,
        "spearman": spearman,
        "error_mean": float(np.mean(errors)),
        "error_std": float(np.std(errors)),
        "abs_error_p50": float(np.quantile(np.abs(errors), 0.50)),
        "abs_error_p90": float(np.quantile(np.abs(errors), 0.90)),
    }


def log_regression_summary(split_name, targets, predictions, loss=None):
    targets = np.asarray(targets, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    metrics = compute_regression_metrics(targets, predictions)
    baseline_predictions = np.full_like(targets, regression_target_mean, dtype=float)
    baseline_metrics = compute_regression_metrics(targets, baseline_predictions)
    loss_text = f", Loss={loss:.4f}" if loss is not None else ""
    logger.info(
        f"{split_name} regression metrics: RMSE={metrics['rmse']:.4f}, "
        f"MAE={metrics['mae']:.4f}, R2={metrics['r2']:.4f}, "
        f"Pearson={metrics['pearson']:.4f}, Spearman={metrics['spearman']:.4f}"
        f"{loss_text}"
    )
    logger.info(
        f"{split_name} mean-baseline metrics: RMSE={baseline_metrics['rmse']:.4f}, "
        f"MAE={baseline_metrics['mae']:.4f}, R2={baseline_metrics['r2']:.4f}"
    )
    logger.info(
        f"{split_name} target summary: mean={targets.mean():.6f}, "
        f"std={targets.std():.6f}, min={targets.min():.6f}, max={targets.max():.6f}"
    )
    logger.info(
        f"{split_name} prediction summary: mean={predictions.mean():.6f}, "
        f"std={predictions.std():.6f}, min={predictions.min():.6f}, "
        f"max={predictions.max():.6f}"
    )
    logger.info(
        f"{split_name} error summary: mean={metrics['error_mean']:.6f}, "
        f"std={metrics['error_std']:.6f}, "
        f"abs_error_p50={metrics['abs_error_p50']:.6f}, "
        f"abs_error_p90={metrics['abs_error_p90']:.6f}"
    )
    return metrics, baseline_metrics


def log_model_mode_diagnostics(trainer, split_name, loader):
    configs = [
        ("train", True, "train_mode_dropout_disabled"),
        ("train", False, "train_mode_dropout_active"),
    ]
    for model_mode, disable_dropout, diagnostic_name in configs:
        results = trainer.evaluate(
            loader,
            generate_images=False,
            model_mode=model_mode,
            disable_dropout=disable_dropout,
        )
        if trainer.task == "Classification":
            accuracy, _, _, f1, roc_auc, _, targets, probabilities, _, binarized = results
            raw_logits = getattr(trainer, "last_eval_logits", probabilities)
            if len(raw_logits) != len(probabilities):
                raw_logits = probabilities
            logger.info(
                f"{split_name} mode diagnostic [{diagnostic_name}] at threshold "
                f"{trainer.threshold:.6f}: F1={f1:.4f}, Accuracy={accuracy:.4f}, "
                f"ROC_AUC={roc_auc:.4f}"
            )
        else:
            rmse, loss, targets, predictions, _ = results
            targets = inverse_regression_values(targets)
            predictions = inverse_regression_values(predictions)
            metrics = compute_regression_metrics(targets, predictions)
            logger.info(
                f"{split_name} mode diagnostic [{diagnostic_name}]: "
                f"RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, "
                f"R2={metrics['r2']:.4f}, Pearson={metrics['pearson']:.4f}, "
                f"Spearman={metrics['spearman']:.4f}, Loss={loss:.4f}"
            )


# Config
batch_size = 32
epochs = 200
hidden_channels = 64
dropout_rate = 0.2
lr = 1e-3
node_block = "ABMP"
normalization = "graphnorm"
task = "Classification"
target_column = "Actividad"
standardize_regression_target = True
calibration_fraction = 0.15
random_eval_fraction = 0.2
threshold_metric = "f1"
message_passing_steps = 1
auto_message_passing = True
max_message_passing_steps = 4
message_passing_min_delta = 1e-4
message_passing_patience = 1
preprocessing_num_workers = min(128, os.cpu_count() or 1)
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "evaluate_outputs")
os.makedirs(output_dir, exist_ok=True)
preprocessing_cache_dir = os.path.abspath(
    os.path.join(output_dir, "processed_molecule_cache")
)


dataset_path = os.path.join(script_dir, '../data/bace.csv')
dataset = pd.read_csv(dataset_path)
required_columns = {"SMILES", target_column, "Title"}
missing_columns = required_columns.difference(dataset.columns)
if missing_columns:
    raise ValueError(f"Missing required columns in dataset: {sorted(missing_columns)}")

if task not in {"Classification", "Regression"}:
    raise ValueError("task must be either 'Classification' or 'Regression'.")

split_description = ""
if "Model" in dataset.columns:
    model_split = dataset["Model"].astype(str).str.strip().str.lower()
    train_model_values = ["test", "valid"]
    eval_model_values = ["train"]
    train_rows = dataset[model_split.isin(train_model_values)].copy()
    eval_rows = dataset[model_split.isin(eval_model_values)].copy()
    split_description = (
        f"Model == {'+'.join(train_model_values)} for training and "
        f"Model == {'+'.join(eval_model_values)} for evaluation"
    )
    logger.info(
        "Dataset Model split counts before preprocessing: "
        f"{dataset['Model'].value_counts(dropna=False).to_dict()}"
    )
    logger.info(
        f"Using Model == Test + Valid for training: {len(train_rows)} molecules. "
        f"Using Model == Train for evaluation: {len(eval_rows)} molecules."
    )
else:
    stratify_labels = None
    if task == "Classification":
        label_values, label_counts = np.unique(dataset[target_column], return_counts=True)
        if len(label_values) > 1 and int(label_counts.min()) >= 2:
            stratify_labels = dataset[target_column]
            logger.info(
                "Model column not found; using stratified random split by labels."
            )
        else:
            logger.warning(
                "Model column not found, but stratified split is not possible because "
                "the classification target has fewer than two samples in at least one class. "
                "Using non-stratified random split."
            )
    else:
        logger.info(
            "Model column not found; using non-stratified random split for regression."
        )
    train_rows, eval_rows = train_test_split(
        dataset,
        test_size=random_eval_fraction,
        random_state=42,
        shuffle=True,
        stratify=stratify_labels,
    )
    train_rows = train_rows.copy()
    eval_rows = eval_rows.copy()
    split_description = (
        f"random split with train={1.0 - random_eval_fraction:.2f}, "
        f"evaluation={random_eval_fraction:.2f}"
    )
    logger.info(
        f"Random split before preprocessing: training={len(train_rows)} molecules, "
        f"evaluation={len(eval_rows)} molecules."
    )

if task == "Classification":
    logger.info(
        "Training labels before preprocessing "
        f"({split_description}): "
        f"negatives={int((train_rows[target_column] == 0).sum())}, "
        f"positives={int((train_rows[target_column] == 1).sum())}"
    )
    logger.info(
        "Evaluation labels before preprocessing "
        f"({split_description}): "
        f"negatives={int((eval_rows[target_column] == 0).sum())}, "
        f"positives={int((eval_rows[target_column] == 1).sum())}"
    )
else:
    logger.info(
        f"Training target before preprocessing ({target_column}): "
        f"count={len(train_rows)}, mean={train_rows[target_column].mean():.6f}, "
        f"std={train_rows[target_column].std():.6f}, "
        f"min={train_rows[target_column].min():.6f}, "
        f"max={train_rows[target_column].max():.6f}"
    )
    logger.info(
        f"Evaluation target before preprocessing ({target_column}): "
        f"count={len(eval_rows)}, mean={eval_rows[target_column].mean():.6f}, "
        f"std={eval_rows[target_column].std():.6f}, "
        f"min={eval_rows[target_column].min():.6f}, "
        f"max={eval_rows[target_column].max():.6f}"
    )
logger.info(f"Processed molecule cache directory: {preprocessing_cache_dir}")
logger.info(
    f"Run configuration: task={task}, target_column={target_column}, "
    f"node_block={node_block}, normalization={normalization}, "
    f"hidden_channels={hidden_channels}, dropout_rate={dropout_rate}, lr={lr}, "
    f"batch_size={batch_size}, epochs={epochs}, "
    f"calibration_fraction={calibration_fraction}, random_eval_fraction={random_eval_fraction}, "
    f"threshold_metric={threshold_metric}, "
    f"message_passing_steps={message_passing_steps}, "
    f"auto_message_passing={auto_message_passing}, "
    f"max_message_passing_steps={max_message_passing_steps}, "
    f"message_passing_min_delta={message_passing_min_delta}, "
    f"message_passing_patience={message_passing_patience}, "
    f"standardize_regression_target={standardize_regression_target}"
)

if train_rows.empty:
    raise ValueError(f"No rows found for training split: {split_description}.")
if eval_rows.empty:
    raise ValueError(f"No rows found for evaluation split: {split_description}.")

smiles_train = train_rows['SMILES'].tolist()
labels_train = train_rows[target_column].tolist()
names_train = train_rows['Title'].tolist()
smiles_eval = eval_rows['SMILES'].tolist()
labels_eval = eval_rows[target_column].tolist()
names_eval = eval_rows['Title'].tolist()

regression_target_mean = 0.0
regression_target_std = 1.0
if task == "Regression":
    labels_train_array = np.asarray(labels_train, dtype=float)
    regression_target_mean = float(labels_train_array.mean())
    regression_target_std = float(labels_train_array.std())
    if regression_target_std <= 0:
        logger.warning(
            "Regression target standard deviation is zero; disabling target scaling."
        )
        regression_target_std = 1.0
    if standardize_regression_target:
        labels_train = [
            (float(label) - regression_target_mean) / regression_target_std
            for label in labels_train
        ]
        labels_eval = [
            (float(label) - regression_target_mean) / regression_target_std
            for label in labels_eval
        ]
        logger.info(
            f"Regression target standardization enabled: "
            f"mean={regression_target_mean:.6f}, std={regression_target_std:.6f}"
        )

# === Dataset Preparation ===
train_dataset = MolecularDataset(
    smiles_train,
    names_train,
    labels_train,
    node_block=node_block,
    num_workers=preprocessing_num_workers,
    cache_dir=preprocessing_cache_dir,
)
eval_dataset = MolecularDataset(
    smiles_eval,
    names_eval,
    labels_eval,
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
    raise ValueError("No successful labels.")

if task == "Classification":
    logger.info(
        f"Training labels after preprocessing: "
        f"negatives={successful_labels_train.count(0)}, positives={successful_labels_train.count(1)}"
    )
    logger.info(
        f"Evaluation labels after preprocessing: "
        f"negatives={eval_dataset.successful_labels.count(0)}, positives={eval_dataset.successful_labels.count(1)}"
    )
else:
    train_targets_array = inverse_regression_values(successful_labels_train)
    eval_targets_array = inverse_regression_values(eval_dataset.successful_labels)
    logger.info(
        f"Training target after preprocessing: count={len(train_targets_array)}, "
        f"mean={train_targets_array.mean():.6f}, std={train_targets_array.std():.6f}, "
        f"min={train_targets_array.min():.6f}, max={train_targets_array.max():.6f}"
    )
    logger.info(
        f"Evaluation target after preprocessing: count={len(eval_targets_array)}, "
        f"mean={eval_targets_array.mean():.6f}, std={eval_targets_array.std():.6f}, "
        f"min={eval_targets_array.min():.6f}, max={eval_targets_array.max():.6f}"
    )

all_train_indices = np.arange(len(train_dataset))
label_values, label_counts = np.unique(successful_labels_train, return_counts=True)
can_stratify = task == "Classification" and len(label_values) > 1 and label_counts.min() >= 2
if 0.0 < calibration_fraction < 1.0 and len(all_train_indices) > 1:
    split_stratify = None
    if task == "Classification" and can_stratify:
        split_stratify = np.asarray(successful_labels_train)
    elif task == "Classification":
        logger.warning(
            "Fit/calibration split cannot be stratified because at least one class "
            "has fewer than two processed molecules. Using a random split."
        )
    fit_indices, calibration_indices = train_test_split(
        all_train_indices,
        test_size=calibration_fraction,
        random_state=42,
        stratify=split_stratify,
    )
else:
    fit_indices = all_train_indices
    calibration_indices = np.array([], dtype=int)

fit_indices = fit_indices.tolist()
calibration_indices = calibration_indices.tolist()
fit_labels = [successful_labels_train[i] for i in fit_indices]
fit_smiles = [successful_smiles_train[i] for i in fit_indices]
fit_names = [successful_names_train[i] for i in fit_indices]

if task == "Classification":
    logger.info(
        f"Fit/calibration split after preprocessing: fit={len(fit_indices)} "
        f"(negatives={fit_labels.count(0)}, positives={fit_labels.count(1)}), "
        f"calibration={len(calibration_indices)} "
        f"(negatives={sum(successful_labels_train[i] == 0 for i in calibration_indices)}, "
        f"positives={sum(successful_labels_train[i] == 1 for i in calibration_indices)})"
    )
else:
    fit_labels_original = inverse_regression_values(fit_labels)
    if calibration_indices:
        calibration_labels_original = inverse_regression_values(
            [successful_labels_train[i] for i in calibration_indices]
        )
        logger.info(
            f"Fit/validation split after preprocessing for regression: "
            f"fit={len(fit_indices)}, validation={len(calibration_indices)}, "
            f"fit_target_mean={np.mean(fit_labels_original):.6f}, "
            f"validation_target_mean={np.mean(calibration_labels_original):.6f}"
        )
    else:
        logger.info(
            f"Regression uses all preprocessed training molecules for fitting: "
            f"fit={len(fit_indices)}, target_mean={np.mean(fit_labels_original):.6f}"
        )


fit_dataset = Subset(train_dataset, fit_indices)
calibration_dataset = Subset(train_dataset, calibration_indices)

train_loader = GeometricDataLoader(fit_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
train_eval_loader = GeometricDataLoader(train_dataset, batch_size=batch_size, shuffle=False)
calibration_loader = GeometricDataLoader(calibration_dataset, batch_size=batch_size, shuffle=False)
eval_loader = GeometricDataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

# === Trainer Setup ===
trainer = GNNTrainer(
    smiles_list=fit_smiles,
    labels=fit_labels,
    names_list=fit_names,
    hidden_channels=hidden_channels,
    num_node_features=num_node_features,
    global_dim=global_dim,
    lr=lr,
    edge_dim=edge_dim,
    batch_size=batch_size,
    node_block=node_block,
    dropout_rate=dropout_rate,
    input_dir=output_dir,
    task=task,
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

# === Train Model ===
if auto_message_passing:
    logger.info(
        f"Starting automatic message-passing search for {epochs} epochs per candidate."
    )
    search_result = trainer.fit_with_message_passing_search(
        train_loader,
        calibration_loader,
        epochs=epochs,
        regression_target_mean=regression_target_mean,
        regression_target_std=regression_target_std,
    )
    logger.info(
        f"Automatic message-passing search selected "
        f"message_passing_steps={search_result['best_steps']} with "
        f"{search_result['metric_name']}={search_result.get('best_metric', float('nan')):.6f}"
    )
else:
    trainer.setup_model()
    logger.info(f"Starting training for {epochs} epochs.")
    for epoch in range(epochs):
        loss = trainer.train(train_loader)
        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

if task == "Classification" and len(calibration_indices) > 0:
    selected_threshold, threshold_metrics = trainer.choose_threshold(
        calibration_loader,
        metric=threshold_metric,
    )
    logger.info(
        f"Selected threshold from calibration set: threshold={selected_threshold:.6f}, "
        f"F1={threshold_metrics['f1']:.4f}, "
        f"Accuracy={threshold_metrics['accuracy']:.4f}, "
        f"Precision={threshold_metrics['precision']:.4f}, "
        f"Recall={threshold_metrics['recall']:.4f}"
    )
elif task == "Classification":
    logger.warning(
        "Calibration split is empty; keeping existing threshold "
        f"{trainer.threshold:.6f}."
    )

eval_results = trainer.evaluate(
    eval_loader,
    generate_images=True
)
if task == "Classification":
    accuracy, _, _, f1, roc_auc, _, all_targets, all_preds, compound_names, all_preds_binarized = eval_results

    # === Classification Report ===
    raw_logits = getattr(trainer, "last_eval_logits", all_preds)
    if len(raw_logits) != len(all_preds):
        raw_logits = all_preds
    logger.info(
        f"Evaluation-set metrics at threshold {trainer.threshold:.6f}: "
        f"F1={f1:.4f}, Accuracy={accuracy:.4f}, ROC_AUC={roc_auc:.4f}"
    )

    log_model_mode_diagnostics(trainer, "Evaluation set", eval_loader)
    for name, true_label, logit, probability, pred_label in zip(
        compound_names,
        all_targets,
        raw_logits,
        all_preds,
        all_preds_binarized,
    ):
        logger.info(
            f'Compound: {name}, True Label: {true_label}, Raw Logit: {logit:.8f}, '
            f'Probability: {probability:.8f}, '
            f'Probability Threshold: {trainer.threshold:.6f}, '
            f'Predicted Label: {pred_label}'
        )
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    cm = confusion_matrix(all_targets, all_preds_binarized)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix for Evaluation Set")
    plt.savefig(os.path.join(output_dir, "conf_matrix.png"), dpi=600)

    fpr, tpr, _ = roc_curve(all_targets, all_preds)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Evaluation Set')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, "roc_auc_blind_set.png"), dpi=600)
else:
    train_results = trainer.evaluate(
        train_eval_loader,
        generate_images=False
    )
    train_rmse_scaled, train_loss, train_targets_scaled, train_preds_scaled, _ = train_results
    train_targets = inverse_regression_values(train_targets_scaled)
    train_preds = inverse_regression_values(train_preds_scaled)
    train_metrics, _ = log_regression_summary(
        "Training set",
        train_targets,
        train_preds,
        loss=train_loss,
    )

    rmse_scaled, eval_loss, all_targets_scaled, all_preds_scaled, compound_names = eval_results
    all_targets = inverse_regression_values(all_targets_scaled)
    all_preds = inverse_regression_values(all_preds_scaled)
    eval_metrics, baseline_metrics = log_regression_summary(
        "Evaluation set",
        all_targets,
        all_preds,
        loss=eval_loss,
    )

    log_model_mode_diagnostics(trainer, "Evaluation set", eval_loader)
    for name, true_value, prediction in zip(compound_names, all_targets, all_preds):
        logger.info(
            f"Compound: {name}, True Value: {true_value:.8f}, "
            f"Predicted Value: {prediction:.8f}, Error: {prediction - true_value:.8f}"
        )
    print(f"RMSE: {eval_metrics['rmse']:.4f}")
    print(f"MAE: {eval_metrics['mae']:.4f}")
    print(f"R2: {eval_metrics['r2']:.4f}")
    print(f"Pearson: {eval_metrics['pearson']:.4f}")
    print(f"Spearman: {eval_metrics['spearman']:.4f}")
    print(f"Mean-baseline RMSE: {baseline_metrics['rmse']:.4f}")

    plt.figure()
    plt.scatter(all_targets, all_preds, alpha=0.75)
    min_value = min(min(all_targets), min(all_preds))
    max_value = max(max(all_targets), max(all_preds))
    plt.plot([min_value, max_value], [min_value, max_value], color='navy', linestyle='--')
    plt.xlabel("True Value")
    plt.ylabel("Predicted Value")
    plt.title("Regression Predictions for Evaluation Set")
    plt.savefig(os.path.join(output_dir, "regression_predicted_vs_true.png"), dpi=600)
