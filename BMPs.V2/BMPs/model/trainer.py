from .interaction_network import InteractionNetwork
import os
import csv
import logging
import copy
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch.utils.data import WeightedRandomSampler
from BMPs.data.molecular_dataset import MolecularDataset
import time 
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, root_mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import KFold, StratifiedKFold
from matplotlib.colors import LinearSegmentedColormap
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from collections import defaultdict
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import random
seed=42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")
class GNNTrainer:
    def __init__(self, smiles_list, labels=None, names_list=None, node_block="ABMP", hidden_channels=64, task="Classification", num_node_features=5, global_dim=6, lr=0.001, edge_dim=4, batch_size=32, k_folds=5, dropout_rate=0.5, input_dir = "evaluate_outputs", max_norm = 1, epochs = 50, threshold = 0.5, preprocessing_num_workers=32, preprocessing_num_confs=2, preprocessing_max_isomers=8, preprocessing_save_images=False, preprocessing_cache_dir=None, normalization="batchnorm", standardize_regression_target=True, message_passing_steps=1, auto_message_passing=False, max_message_passing_steps=4, message_passing_min_delta=1e-4, message_passing_patience=1):
        self.smiles_list = smiles_list
        self.labels = labels
        self.names_list = names_list
        self.hidden_channels = hidden_channels
        self.num_node_features = num_node_features  
        self.global_dim = global_dim
        self.lr = lr
        self.task = task
        self.batch_size = batch_size
        self.k_folds = k_folds
        self.epochs = epochs  
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dropout_rate = dropout_rate
        self.edge_dim = edge_dim
        self.threshold = threshold
        self.input_dir = input_dir
        self.node_block = node_block
        self.normalization = normalization
        self.standardize_regression_target = standardize_regression_target
        self.message_passing_steps = int(message_passing_steps)
        self.auto_message_passing = auto_message_passing
        self.max_message_passing_steps = int(max_message_passing_steps)
        self.message_passing_min_delta = message_passing_min_delta
        self.message_passing_patience = int(message_passing_patience)
        self.message_passing_search_history = []
        self.max_norm = max_norm
        self.preprocessing_num_workers = preprocessing_num_workers
        self.preprocessing_num_confs = preprocessing_num_confs
        self.preprocessing_max_isomers = preprocessing_max_isomers
        self.preprocessing_save_images = preprocessing_save_images
        self.preprocessing_cache_dir = (
            preprocessing_cache_dir
            if preprocessing_cache_dir is not None
            else os.path.join(self.input_dir, "processed_molecule_cache")
        )
    def get_preprocessing_kwargs(self):
        return {
            "node_block": self.node_block,
            "num_workers": self.preprocessing_num_workers,
            "num_confs": self.preprocessing_num_confs,
            "max_isomers": self.preprocessing_max_isomers,
            "save_images": self.preprocessing_save_images,
            "cache_dir": self.preprocessing_cache_dir,
        }
    def setup_model(self):
        if self.num_node_features <= 0 or self.edge_dim <= 0 or self.global_dim <= 0:
            raise ValueError(
                "Invalid model feature dimensions: "
                f"num_node_features={self.num_node_features}, "
                f"edge_dim={self.edge_dim}, global_dim={self.global_dim}. "
                "This usually means cached molecule metadata is stale or the dataset "
                "did not process valid molecules."
            )
        self.model = InteractionNetwork(
            self.num_node_features,
            self.edge_dim,
            self.hidden_channels,
            self.global_dim,
            self.dropout_rate,
            self.node_block,
            self.normalization,
            self.message_passing_steps,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if self.task == "Regression":
            self.criterion = nn.MSELoss()
        elif self.task == "Classification":
            num_pos = np.sum(np.array(self.labels) == 1)
            num_neg = np.sum(np.array(self.labels) == 0)
            if num_pos == 0:
                raise ValueError("Classification setup found zero positive labels.")
            pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float, device=self.device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            logger.info(
                f"Classification loss setup: negatives={int(num_neg)}, "
                f"positives={int(num_pos)}, pos_weight={float(pos_weight.item()):.6f}"
            )
        else:
            raise ValueError(f"Unsupported task type: {self.task}. Must be 'Classification' or 'Regression'.")
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5)
        logger.info(f"Model normalization: {self.normalization}")
        logger.info(f"Message passing steps: {self.message_passing_steps}")
        print(f"\nMode: {self.node_block.upper() if isinstance(self.node_block, str) else type(self.node_block).__name__}")
        print(f"  Normalization: {self.normalization}")
        print(f"  Message passing steps: {self.message_passing_steps}")
        total_params = 0
        block_params = {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            count = param.numel()
            total_params += count
            block_name = name.split('.')[0] 
            block_params[block_name] = block_params.get(block_name, 0) + count
        for block, count in block_params.items():
            print(f"  {block}: {count:,} parameters")
        print(f"  ➤ Total trainable parameters: {total_params:,}\n")
    def train(self, train_loader):
        self.model.train()  
        total_loss = 0
        for data in train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            out, _ = self.model(
                data.x, data.edge_index, data.edge_attr, data.u, data.batch
            )
            predictions = out.view(-1)
            targets = data.y.view(-1).float()
            if self.task == "Classification":
                loss = self.criterion(predictions, targets)
            elif self.task == "Regression":
                loss = self.criterion(predictions, targets)
            else:
                raise ValueError(f"Unsupported task type: {self.task}")
            loss.backward() 
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)
            self.optimizer.step()
            total_loss += loss.item() * data.num_graphs
        return total_loss / len(train_loader.dataset) 

    def find_best_threshold(self, targets, probabilities, metric="f1"):
        if self.task != "Classification":
            raise ValueError("Automatic threshold selection is only valid for classification.")
        targets = np.asarray(targets, dtype=int)
        probabilities = np.asarray(probabilities, dtype=float)
        if targets.size == 0:
            raise ValueError("Cannot choose a threshold without targets.")
        if len(np.unique(targets)) < 2:
            raise ValueError("Cannot choose a threshold when targets contain only one class.")
        metric = metric.lower()
        unique_probs = np.unique(probabilities)
        if unique_probs.size > 1:
            midpoints = (unique_probs[:-1] + unique_probs[1:]) / 2.0
            candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], midpoints)))
        else:
            candidates = np.array([0.0, 0.5, 1.0])
        best_threshold = 0.5
        best_metrics = None
        best_score = -np.inf
        for threshold in candidates:
            predictions = (probabilities > threshold).astype(int)
            metrics = {
                "accuracy": accuracy_score(targets, predictions),
                "precision": precision_score(targets, predictions, zero_division=0),
                "recall": recall_score(targets, predictions, zero_division=0),
                "f1": f1_score(targets, predictions, zero_division=0),
                "positive_predictions": int(predictions.sum()),
                "total_predictions": int(predictions.size),
            }
            if metric not in metrics:
                raise ValueError(f"Unsupported threshold metric: {metric}")
            score = metrics[metric]
            if (
                score > best_score
                or (
                    score == best_score
                    and abs(float(threshold) - 0.5) < abs(float(best_threshold) - 0.5)
                )
            ):
                best_score = score
                best_threshold = float(threshold)
                best_metrics = metrics
        return best_threshold, best_metrics

    def choose_threshold(self, loader, metric="f1"):
        previous_threshold = self.threshold
        results = self.evaluate(loader, generate_images=False)
        targets = results[6]
        probabilities = results[7]
        threshold, metrics = self.find_best_threshold(targets, probabilities, metric)
        self.threshold = threshold
        logger.info(
            f"Automatic threshold selected on calibration set: "
            f"metric={metric}, previous_threshold={previous_threshold:.6f}, "
            f"selected_threshold={threshold:.6f}, "
            f"F1={metrics['f1']:.4f}, Accuracy={metrics['accuracy']:.4f}, "
            f"Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, "
            f"positive_predictions={metrics['positive_predictions']}/"
            f"{metrics['total_predictions']}"
        )
        return threshold, metrics

    def regression_metrics(self, targets, predictions):
        targets = np.asarray(targets, dtype=float)
        predictions = np.asarray(predictions, dtype=float)
        metrics = {
            "rmse": root_mean_squared_error(targets, predictions),
            "mae": mean_absolute_error(targets, predictions),
            "r2": r2_score(targets, predictions) if len(targets) > 1 else float("nan"),
            "pearson": float("nan"),
            "spearman": float("nan"),
        }
        if len(targets) > 1 and np.std(targets) > 0 and np.std(predictions) > 0:
            metrics["pearson"] = float(np.corrcoef(targets, predictions)[0, 1])
            metrics["spearman"] = float(
                np.corrcoef(np.argsort(np.argsort(targets)), np.argsort(np.argsort(predictions)))[0, 1]
            )
        return metrics

    def _message_passing_metric_improved(self, current, reference):
        if reference is None:
            return True
        if self.task == "Classification":
            return current > reference + self.message_passing_min_delta
        if self.task == "Regression":
            return current < reference - self.message_passing_min_delta
        raise ValueError(f"Unsupported task type: {self.task}")

    def _message_passing_metric_value(
        self,
        loader,
        regression_target_mean=0.0,
        regression_target_std=1.0,
    ):
        results = self.evaluate(loader, generate_images=False)
        if self.task == "Classification":
            accuracy, precision, recall, f1, roc_auc, loss, targets, probabilities, _, _ = results
            comparable_value = roc_auc if not np.isnan(roc_auc) else -np.inf
            return {
                "metric_name": "AUROC",
                "metric_value": comparable_value,
                "display_value": roc_auc,
                "loss": loss,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        if self.task == "Regression":
            rmse, loss, targets, predictions, _ = results
            targets = np.asarray(targets, dtype=float)
            predictions = np.asarray(predictions, dtype=float)
            if self.standardize_regression_target:
                targets = targets * regression_target_std + regression_target_mean
                predictions = predictions * regression_target_std + regression_target_mean
            metrics = self.regression_metrics(targets, predictions)
            return {
                "metric_name": "MAE",
                "metric_value": metrics["mae"],
                "display_value": metrics["mae"],
                "loss": loss,
                **metrics,
            }
        raise ValueError(f"Unsupported task type: {self.task}")

    def fit_with_message_passing_search(
        self,
        train_loader,
        validation_loader,
        epochs=None,
        regression_target_mean=0.0,
        regression_target_std=1.0,
        max_message_passing_steps=None,
        verbose_prefix="",
    ):
        if epochs is None:
            epochs = self.epochs
        max_steps = int(max_message_passing_steps or self.max_message_passing_steps)
        max_steps = max(1, max_steps)
        patience = max(1, self.message_passing_patience)
        if validation_loader is None or len(validation_loader.dataset) == 0:
            logger.warning(
                "Automatic message-passing search requested without a validation "
                "loader. Training with message_passing_steps=%s.",
                self.message_passing_steps,
            )
            self.setup_model()
            for epoch in range(epochs):
                loss = self.train(train_loader)
                logger.info(
                    f"{verbose_prefix}Epoch {epoch + 1}/{epochs}, "
                    f"message_passing_steps={self.message_passing_steps}, Loss={loss:.4f}"
                )
                print(
                    f"{verbose_prefix}Epoch {epoch + 1}/{epochs}, "
                    f"message_passing_steps={self.message_passing_steps}, Loss={loss:.4f}"
                )
            self.message_passing_search_history = []
            return {
                "best_steps": self.message_passing_steps,
                "history": [],
                "metric_name": "unavailable",
            }

        best_steps = None
        best_state = None
        best_metric = None
        previous_metric = None
        unimproved_steps = 0
        history = []
        original_steps = self.message_passing_steps
        logger.info(
            f"Starting automatic message-passing search: task={self.task}, "
            f"candidate_steps=1..{max_steps}, min_delta={self.message_passing_min_delta}, "
            f"patience={patience}"
        )
        for steps in range(1, max_steps + 1):
            self.message_passing_steps = steps
            self.setup_model()
            logger.info(f"Training candidate message_passing_steps={steps}.")
            print(f"{verbose_prefix}Training message_passing_steps={steps}")
            for epoch in range(epochs):
                loss = self.train(train_loader)
                logger.info(
                    f"{verbose_prefix}Epoch {epoch + 1}/{epochs}, "
                    f"message_passing_steps={steps}, Loss={loss:.4f}"
                )
                print(
                    f"{verbose_prefix}Epoch {epoch + 1}/{epochs}, "
                    f"message_passing_steps={steps}, Loss={loss:.4f}"
                )

            metric = self._message_passing_metric_value(
                validation_loader,
                regression_target_mean=regression_target_mean,
                regression_target_std=regression_target_std,
            )
            metric_value = metric["metric_value"]
            improved_vs_previous = self._message_passing_metric_improved(
                metric_value,
                previous_metric,
            )
            improved_vs_best = self._message_passing_metric_improved(
                metric_value,
                best_metric,
            )
            history_item = {
                "message_passing_steps": steps,
                "improved_vs_previous": improved_vs_previous,
                **metric,
            }
            history.append(history_item)
            logger.info(
                f"Message-passing candidate result: steps={steps}, "
                f"{metric['metric_name']}={metric['display_value']:.6f}, "
                f"improved_vs_previous={improved_vs_previous}."
            )
            print(
                f"{verbose_prefix}message_passing_steps={steps}: "
                f"{metric['metric_name']}={metric['display_value']:.4f}, "
                f"improved_vs_previous={improved_vs_previous}"
            )
            if improved_vs_best:
                best_steps = steps
                best_metric = metric_value
                best_state = copy.deepcopy(self.model.state_dict())
            if previous_metric is not None and not improved_vs_previous:
                unimproved_steps += 1
            else:
                unimproved_steps = 0
            previous_metric = metric_value
            if previous_metric is not None and unimproved_steps >= patience:
                logger.info(
                    f"Stopping message-passing search after {steps} steps because "
                    f"performance did not improve for {unimproved_steps} candidate(s)."
                )
                break

        if best_state is None:
            self.message_passing_steps = original_steps
            self.setup_model()
            raise ValueError("Automatic message-passing search did not train a valid model.")
        self.message_passing_steps = best_steps
        self.setup_model()
        self.model.load_state_dict(best_state)
        self.message_passing_search_history = history
        metric_name = history[0]["metric_name"] if history else "metric"
        logger.info(
            f"Selected message_passing_steps={best_steps} with best {metric_name}={best_metric:.6f}."
        )
        print(
            f"{verbose_prefix}Selected message_passing_steps={best_steps} "
            f"with best {metric_name}={best_metric:.4f}"
        )
        return {
            "best_steps": best_steps,
            "best_metric": best_metric,
            "metric_name": metric_name,
            "history": history,
        }

    def normalize_atom_importance_for_image(
        self,
        atom_scores,
        lower_quantile=0.05,
        upper_quantile=0.95,
        protected_threshold=0.70,
        editable_threshold=0.30,
    ):
        atom_scores = atom_scores.view(-1).float()
        raw_scores = atom_scores.detach().clone()
        if atom_scores.numel() == 0:
            empty_mask = atom_scores.bool()
            return atom_scores, raw_scores, atom_scores, empty_mask, empty_mask
        atom_scores = torch.clamp(atom_scores, min=0)
        transformed_scores = torch.log1p(atom_scores)
        if transformed_scores.numel() == 1:
            visual_scores = torch.ones_like(transformed_scores)
            protected_mask = visual_scores >= protected_threshold
            editable_mask = visual_scores <= editable_threshold
            return visual_scores, raw_scores, transformed_scores, protected_mask, editable_mask
        low = torch.quantile(transformed_scores, lower_quantile)
        high = torch.quantile(transformed_scores, upper_quantile)
        if high <= low:
            min_val = transformed_scores.min()
            max_val = transformed_scores.max()
            if max_val > min_val:
                visual_scores = (transformed_scores - min_val) / (max_val - min_val)
            else:
                visual_scores = torch.zeros_like(transformed_scores)
        else:
            clipped = torch.clamp(transformed_scores, min=low, max=high)
            visual_scores = (clipped - low) / (high - low)
        visual_scores = torch.clamp(visual_scores, 0, 1)
        protected_mask = visual_scores >= protected_threshold
        editable_mask = visual_scores <= editable_threshold
        return visual_scores, raw_scores, transformed_scores, protected_mask, editable_mask

    def save_molecule_image_with_importance(self, mol, name, atom_importance, cmap, output_dir="molecule_images", img_size=(400, 400)):
        os.makedirs(output_dir, exist_ok=True)
        drawer = rdMolDraw2D.MolDraw2DCairo(img_size[0], img_size[1])
        dos = drawer.drawOptions()
        dos.addStereoAnnotation = True
        dos.bondLineWidth = 2  
        dos.circleAtoms = True  
        #dos.FontSize = 1 
        heteroatoms = {7, 8, 9, 16, 17, 35, 53}  
        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            atom_num = atom.GetAtomicNum()
            if atom_num in heteroatoms:
                dos.atomLabels[atom_idx] = atom.GetSymbol()  
                dos.annotationFontScale = 1.2
            else:
                dos.atomLabels[atom_idx] = ""  
        highlight_atoms = defaultdict(list)
        atom_rads = {}
        width_mults = {}
        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            importance_value = float(atom_importance[atom_idx]) 
            importance_value = max(0.0, min(1.0, importance_value)) 
            color = cmap(int(importance_value * 255))[:3] 
            highlight_atoms[atom_idx].append(color)
            atom_rads[atom_idx] = 0.2 + 0.5 * importance_value
        try:
            drawer.DrawMoleculeWithHighlights(
                mol,
                legend='',  
                highlight_atom_map=dict(highlight_atoms),  
                highlight_bond_map={}, 
                highlight_radii=atom_rads,  
                highlight_linewidth_multipliers=width_mults,  
                confId=-1  
            )
        except Exception as e:
            print(f"Error drawing molecule highlights: {e}")
            raise e
        mol_img_path = os.path.join(output_dir, f"molecule_{name}_importance.png")
        with open(mol_img_path, 'wb') as img_file:
            img_file.write(drawer.GetDrawingText())
        fig, ax = plt.subplots(figsize=(2, 4))  
        fig.subplots_adjust(left=0.9, right=0.95, top=0.8, bottom=0)  
        norm = plt.Normalize(vmin=0, vmax=1)
        colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax)
        colorbar.ax.tick_params(labelsize=10) 
        colorbar.set_label("Node Prediction", fontsize=13)  
        colorbar_img_path = os.path.join(output_dir, f"colorbar_{name}.png")
        plt.savefig(colorbar_img_path, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        mol_img = Image.open(mol_img_path)
        colorbar_img = Image.open(colorbar_img_path)
        combined_width = mol_img.width + colorbar_img.width + 10  
        combined_height = max(mol_img.height, colorbar_img.height)
        combined_img = Image.new('RGB', (combined_width, combined_height), (255, 255, 255))
        combined_img.paste(mol_img, (0, 0))
        combined_img.paste(colorbar_img, (mol_img.width + 10, 0)) 
        final_img_path = os.path.join(output_dir, f"{name}.png")
        combined_img.save(final_img_path)
        os.remove(mol_img_path)
        os.remove(colorbar_img_path)
        return final_img_path
    def process_and_save_molecule_images(self, data, atom_importance, cmap, molecule_cache):
        atom_index = 0        
        for i, mol_data in enumerate(data.to_data_list()):
            if mol_data.smiles in molecule_cache:
                mol = molecule_cache[mol_data.smiles]
            else:
                s = Standardizer()  
                mol = Chem.MolFromSmiles(mol_data.smiles)
                mol = s.standardize(mol)
                mol_with_h = Chem.AddHs(mol)
                Chem.AssignStereochemistryFrom3D(mol_with_h, replaceExistingTags=True)
                mol = Chem.RemoveHs(mol_with_h)
                Chem.SanitizeMol(mol)                
                molecule_cache[mol_data.smiles] = mol    
            name = mol_data.name  
            num_atoms = mol.GetNumAtoms()
            assert atom_index + num_atoms <= len(atom_importance), \
                f"Atom index out of range: {atom_index + num_atoms} > {len(atom_importance)}"    
            mol_atom_importance = atom_importance[atom_index:atom_index + num_atoms]
            self.save_molecule_image_with_importance(mol, name, mol_atom_importance, cmap=cmap)
            atom_index += num_atoms
    def evaluate(
        self,
        loader,
        generate_images=True,
        log_raw_predictions=False,
        model_mode="eval",
        disable_dropout=False,
    ):
        module_states = [(module, module.training) for module in self.model.modules()]
        batchnorm_momentums = []
        if model_mode == "eval":
            self.model.eval()
        elif model_mode == "train":
            self.model.train()
            for module in self.model.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    batchnorm_momentums.append((module, module.momentum))
                    module.momentum = 0.0
                if disable_dropout and isinstance(module, nn.Dropout):
                    module.eval()
        else:
            raise ValueError("model_mode must be 'eval' or 'train'.")
        if not hasattr(self, 'cached_cmap'):
            colors = [
                (225, 255, 255),  
                (139, 124, 210),     
                (202, 21, 109),   
                (254, 97, 0),  
            ]
            colors = [tuple(c / 255.0 for c in color) for color in colors]
            self.cached_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
        cmap = self.cached_cmap
        total_loss = 0
        all_targets = []
        all_preds = []
        all_logits = []
        compound_names = []  
        correct = 0  
        for data in loader: 
            data = data.to(self.device)
            with torch.no_grad():
                out, atom_importance = self.model(data.x, data.edge_index, data.edge_attr, data.u, data.batch)
                predictions = out.view(-1)
                targets = data.y.view(-1).float()
                loss = self.criterion(predictions, targets)
                total_loss += loss.item() * data.num_graphs
            data_list = data.to_data_list()
            compound_names.extend([d.name for d in data_list])
            atom_importance_list = atom_importance.detach().cpu().split([d.num_nodes for d in data_list])
            for mol_data, mol_importance in zip(data_list, atom_importance_list):
                mol_name = mol_data.name
                mol = Chem.MolFromSmiles(mol_data.smiles)
                (
                    normalized_importance,
                    raw_atom_scores,
                    log_atom_scores,
                    protected_mask,
                    editable_mask,
                ) = self.normalize_atom_importance_for_image(mol_importance)
                if generate_images:
                    self.save_molecule_image_with_importance(
                        mol, mol_name, normalized_importance, cmap
                    )
                    raw_scores = raw_atom_scores.detach().cpu().numpy()
                    log_scores = log_atom_scores.detach().cpu().numpy()
                    image_scores = normalized_importance.detach().cpu().numpy()
                    protected = protected_mask.detach().cpu().numpy()
                    editable = editable_mask.detach().cpu().numpy()
                    with open(f"{self.input_dir}/{mol_name}_node_scores.txt", "w") as f:
                        for i, (
                            raw_score,
                            log_score,
                            image_score,
                            is_protected,
                            is_editable,
                        ) in enumerate(
                            zip(raw_scores, log_scores, image_scores, protected, editable)
                        ):
                            atom_symbol = mol.GetAtomWithIdx(i).GetSymbol()
                            f.write(
                                f"Atom {i} ({atom_symbol}): "
                                f"raw_message_score={raw_score:.6f}, "
                                f"log1p_score={log_score:.6f}, "
                                f"normalized_score={image_score:.4f}, "
                                f"protected_core={int(is_protected)}, "
                                f"editable_region={int(is_editable)}\n"
                            )
            if self.task == "Classification":
                probs = torch.sigmoid(predictions)
                pred = (probs > self.threshold).float()
                all_logits.extend(predictions.detach().cpu().numpy().flatten())
                all_targets.extend(targets.detach().cpu().numpy().flatten())
                all_preds.extend(probs.detach().cpu().numpy().flatten())
                correct += (pred == targets).sum().item()
            elif self.task == "Regression":
                all_targets.extend(targets.detach().cpu().numpy().flatten())
                all_preds.extend(predictions.detach().cpu().numpy().flatten())
            else:
                raise ValueError(f"Unsupported task type: {self.task}")

        if self.task == "Classification":
            all_preds_binarized = [1 if prob > self.threshold else 0 for prob in all_preds]
            self.last_eval_logits = np.array(all_logits)
            if len(all_logits) > 0:
                mode_description = model_mode
                if model_mode == "train" and disable_dropout:
                    mode_description = "train_batchnorm_dropout_disabled"
                logger.info(
                    f"Evaluation raw prediction summary ({mode_description}): "
                    f"logit_min={np.min(all_logits):.6f}, "
                    f"logit_max={np.max(all_logits):.6f}, "
                    f"logit_mean={np.mean(all_logits):.6f}, "
                    f"probability_min={np.min(all_preds):.6f}, "
                    f"probability_max={np.max(all_preds):.6f}, "
                    f"probability_mean={np.mean(all_preds):.6f}, "
                    f"probability_threshold={self.threshold:.6f}, "
                    f"positive_predictions={sum(all_preds_binarized)}/{len(all_preds_binarized)}."
                )
            if log_raw_predictions:
                for name, true_label, logit, prob, pred_label in zip(
                    compound_names,
                    all_targets,
                    all_logits,
                    all_preds,
                    all_preds_binarized,
                ):
                    logger.info(
                        f"Compound: {name}, True Label: {true_label}, "
                        f"Raw Logit: {logit:.8f}, "
                        f"Probability: {prob:.8f}, "
                        f"Probability Threshold: {self.threshold:.4f}, "
                        f"Predicted Label: {pred_label}"
                    )
            accuracy = accuracy_score(all_targets, all_preds_binarized)
            precision = precision_score(all_targets, all_preds_binarized, zero_division=0)
            recall = recall_score(all_targets, all_preds_binarized, zero_division=0)
            f1 = f1_score(all_targets, all_preds_binarized, zero_division=0)
            if len(np.unique(all_targets)) > 1:
                fpr, tpr, _ = roc_curve(all_targets, all_preds)
                roc_auc = auc(fpr, tpr)
            else:
                roc_auc = float("nan")
                logger.warning("ROC AUC is undefined because evaluation targets contain only one class.")
            result = accuracy, precision, recall, f1, roc_auc, total_loss / len(loader.dataset), all_targets, all_preds, compound_names, all_preds_binarized
        elif self.task == "Regression":
            rmse = root_mean_squared_error(all_targets, all_preds)
            result = rmse, total_loss / len(loader.dataset), all_targets, all_preds, compound_names
        else:
            raise ValueError(f"Unsupported task type: {self.task}")
        for module, momentum in batchnorm_momentums:
            module.momentum = momentum
        for module, training in module_states:
            module.train(training)
        return result

    def cross_validate(self):
        label_values, class_counts = (
            np.unique(self.labels, return_counts=True)
            if self.labels is not None
            else (np.array([]), np.array([]))
        )
        can_stratify = (
            self.task == "Classification"
            and len(label_values) > 1
            and len(class_counts) > 0
            and int(np.min(class_counts)) >= self.k_folds
        )
        if can_stratify:
            kf = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=seed)
            splits = kf.split(self.smiles_list, self.labels)
        else:
            if self.task == "Classification":
                logger.warning(
                    "Using non-stratified KFold because at least one class has fewer "
                    f"than k_folds={self.k_folds} samples."
                )
            kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=seed)
            splits = kf.split(self.smiles_list)
        fold_val_losses = []
        fold_train_losses = []
        fold_epoch_times = []
        original_labels = list(self.labels)
        if self.task == "Classification":
            fold_train_accuracies, fold_val_accuracies = [], []
            fold_train_f1s, fold_val_f1s = [], []
            fpr_list, tpr_list, auc_list = [], [], []
        elif self.task == "Regression":
            fold_train_rmses, fold_val_rmses = [], []
            fold_train_maes, fold_val_maes = [], []
            fold_train_r2s, fold_val_r2s = [], []
            fold_val_pearsons, fold_val_spearmans = [], []
        for fold, (train_idx, val_idx) in enumerate(splits):
            start_fold_time = time.time()
            print(f'Fold {fold + 1}/{self.k_folds}')
            smiles_train = [self.smiles_list[i] for i in train_idx]
            names_train = [self.names_list[i] for i in train_idx]
            labels_train = [original_labels[i] for i in train_idx]
            smiles_val = [self.smiles_list[i] for i in val_idx]
            labels_val = [original_labels[i] for i in val_idx]
            names_val = [self.names_list[i] for i in val_idx]
            regression_mean = 0.0
            regression_std = 1.0
            labels_train_original = np.asarray(labels_train, dtype=float)
            labels_val_original = np.asarray(labels_val, dtype=float)
            if self.task == "Regression" and self.standardize_regression_target:
                regression_mean = float(labels_train_original.mean())
                regression_std = float(labels_train_original.std())
                if regression_std <= 0:
                    logger.warning(
                        f"Fold {fold + 1}: regression target std is zero; "
                        "using unscaled targets."
                    )
                    regression_std = 1.0
                labels_train = ((labels_train_original - regression_mean) / regression_std).tolist()
                labels_val = ((labels_val_original - regression_mean) / regression_std).tolist()
                logger.info(
                    f"Fold {fold + 1}: regression target standardization "
                    f"mean={regression_mean:.6f}, std={regression_std:.6f}"
                )
            train_dataset = MolecularDataset(
                smiles_train,
                names_train,
                labels_train,
                **self.get_preprocessing_kwargs()
            )
            val_dataset = MolecularDataset(
                smiles_val,
                names_val,
                labels_val,
                **self.get_preprocessing_kwargs()
            )
            self.global_dim = train_dataset.global_dim
            self.edge_dim = train_dataset.edge_dim
            self.num_node_features = train_dataset.num_node_features  
            train_loader = GeometricDataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
            val_loader = GeometricDataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)    
            self.labels = train_dataset.successful_labels
            if not self.auto_message_passing:
                self.setup_model()
            epoch_train_losses, epoch_val_losses = [], []
            epoch_times = []
            if self.task == "Classification":
                epoch_train_accuracies, epoch_val_accuracies = [], []
                epoch_train_f1s, epoch_val_f1s = [], []
            elif self.task == "Regression":
                epoch_train_rmses, epoch_val_rmses = [], []
                epoch_train_maes, epoch_val_maes = [], []
                epoch_train_r2s, epoch_val_r2s = [], []
                epoch_val_pearsons, epoch_val_spearmans = [], []
            if self.auto_message_passing:
                self.fit_with_message_passing_search(
                    train_loader,
                    val_loader,
                    epochs=self.epochs,
                    regression_target_mean=regression_mean,
                    regression_target_std=regression_std,
                    verbose_prefix=f"Fold {fold + 1} | ",
                )
                epoch_iterable = []
            else:
                epoch_iterable = range(self.epochs)
            for epoch in epoch_iterable:
                start_time = time.time()
                _ = self.train(train_loader)
                if self.task == "Classification":
                    train_acc, _, _, train_f1, _, train_loss, _, _, _, _ = self.evaluate(train_loader, generate_images=False)
                    val_acc, _, _, val_f1, val_auc, val_loss, targets, preds, _, _ = self.evaluate(val_loader, generate_images=False)
                elif self.task == "Regression":
                    _, train_loss, train_targets, train_preds, _ = self.evaluate(train_loader, generate_images=False)
                    _, val_loss, targets, preds, _ = self.evaluate(val_loader, generate_images=False)
                    train_targets = np.asarray(train_targets, dtype=float)
                    train_preds = np.asarray(train_preds, dtype=float)
                    targets = np.asarray(targets, dtype=float)
                    preds = np.asarray(preds, dtype=float)
                    if self.standardize_regression_target:
                        train_targets = train_targets * regression_std + regression_mean
                        train_preds = train_preds * regression_std + regression_mean
                        targets = targets * regression_std + regression_mean
                        preds = preds * regression_std + regression_mean
                    train_metrics = self.regression_metrics(train_targets, train_preds)
                    val_metrics = self.regression_metrics(targets, preds)
                epoch_time = time.time() - start_time
                self.scheduler.step(val_loss) 
                current_lr = self.optimizer.param_groups[0]['lr']
                epoch_train_losses.append(train_loss)
                epoch_val_losses.append(val_loss)
                epoch_times.append(epoch_time)
                if self.task == "Classification":
                    epoch_train_accuracies.append(train_acc)
                    epoch_val_accuracies.append(val_acc)
                    epoch_train_f1s.append(train_f1)
                    epoch_val_f1s.append(val_f1)
                    if len(np.unique(targets)) > 1:
                        fpr, tpr, _ = roc_curve(targets, preds)
                        fpr_list.append(fpr)
                        tpr_list.append(tpr)
                        auc_list.append(val_auc)
                    print(f'Epoch {epoch+1} | Loss: {train_loss:.4f}/{val_loss:.4f} | '
                        f'Acc: {train_acc:.4f}/{val_acc:.4f} | '
                        f'F1: {train_f1:.4f}/{val_f1:.4f} | Time: {epoch_time:.2f}s | LR: {current_lr:.6f}')
                elif self.task == "Regression":
                    epoch_train_rmses.append(train_metrics["rmse"])
                    epoch_val_rmses.append(val_metrics["rmse"])
                    epoch_train_maes.append(train_metrics["mae"])
                    epoch_val_maes.append(val_metrics["mae"])
                    epoch_train_r2s.append(train_metrics["r2"])
                    epoch_val_r2s.append(val_metrics["r2"])
                    epoch_val_pearsons.append(val_metrics["pearson"])
                    epoch_val_spearmans.append(val_metrics["spearman"])
                    print(f'Epoch {epoch+1} | Loss: {train_loss:.4f}/{val_loss:.4f} | '
                        f'RMSE: {train_metrics["rmse"]:.4f}/{val_metrics["rmse"]:.4f} | '
                        f'MAE: {train_metrics["mae"]:.4f}/{val_metrics["mae"]:.4f} | '
                        f'R2: {train_metrics["r2"]:.4f}/{val_metrics["r2"]:.4f} | '
                        f'Time: {epoch_time:.2f}s | LR: {current_lr:.6f}')
            if self.auto_message_passing:
                start_time = time.time()
                if self.task == "Classification":
                    train_acc, _, _, train_f1, _, train_loss, _, _, _, _ = self.evaluate(train_loader, generate_images=False)
                    val_acc, _, _, val_f1, val_auc, val_loss, targets, preds, _, _ = self.evaluate(val_loader, generate_images=False)
                    train_metrics = None
                    val_metrics = None
                elif self.task == "Regression":
                    _, train_loss, train_targets, train_preds, _ = self.evaluate(train_loader, generate_images=False)
                    _, val_loss, targets, preds, _ = self.evaluate(val_loader, generate_images=False)
                    train_targets = np.asarray(train_targets, dtype=float)
                    train_preds = np.asarray(train_preds, dtype=float)
                    targets = np.asarray(targets, dtype=float)
                    preds = np.asarray(preds, dtype=float)
                    if self.standardize_regression_target:
                        train_targets = train_targets * regression_std + regression_mean
                        train_preds = train_preds * regression_std + regression_mean
                        targets = targets * regression_std + regression_mean
                        preds = preds * regression_std + regression_mean
                    train_metrics = self.regression_metrics(train_targets, train_preds)
                    val_metrics = self.regression_metrics(targets, preds)
                epoch_time = time.time() - start_time
                epoch_train_losses.append(train_loss)
                epoch_val_losses.append(val_loss)
                epoch_times.append(epoch_time)
                if self.task == "Classification":
                    epoch_train_accuracies.append(train_acc)
                    epoch_val_accuracies.append(val_acc)
                    epoch_train_f1s.append(train_f1)
                    epoch_val_f1s.append(val_f1)
                    if len(np.unique(targets)) > 1:
                        fpr, tpr, _ = roc_curve(targets, preds)
                        fpr_list.append(fpr)
                        tpr_list.append(tpr)
                        auc_list.append(val_auc)
                    print(f'Fold {fold+1} selected steps={self.message_passing_steps} | '
                        f'Loss: {train_loss:.4f}/{val_loss:.4f} | '
                        f'Acc: {train_acc:.4f}/{val_acc:.4f} | '
                        f'F1: {train_f1:.4f}/{val_f1:.4f} | AUROC: {val_auc:.4f}')
                elif self.task == "Regression":
                    epoch_train_rmses.append(train_metrics["rmse"])
                    epoch_val_rmses.append(val_metrics["rmse"])
                    epoch_train_maes.append(train_metrics["mae"])
                    epoch_val_maes.append(val_metrics["mae"])
                    epoch_train_r2s.append(train_metrics["r2"])
                    epoch_val_r2s.append(val_metrics["r2"])
                    epoch_val_pearsons.append(val_metrics["pearson"])
                    epoch_val_spearmans.append(val_metrics["spearman"])
                    print(f'Fold {fold+1} selected steps={self.message_passing_steps} | '
                        f'Loss: {train_loss:.4f}/{val_loss:.4f} | '
                        f'RMSE: {train_metrics["rmse"]:.4f}/{val_metrics["rmse"]:.4f} | '
                        f'MAE: {train_metrics["mae"]:.4f}/{val_metrics["mae"]:.4f} | '
                        f'R2: {train_metrics["r2"]:.4f}/{val_metrics["r2"]:.4f}')
            fold_train_losses.append(epoch_train_losses)
            fold_val_losses.append(epoch_val_losses)
            fold_epoch_times.append(epoch_times)
            if self.task == "Classification":
                fold_train_accuracies.append(epoch_train_accuracies)
                fold_val_accuracies.append(epoch_val_accuracies)
                fold_train_f1s.append(epoch_train_f1s)
                fold_val_f1s.append(epoch_val_f1s)
            elif self.task == "Regression":
                fold_train_rmses.append(epoch_train_rmses)
                fold_val_rmses.append(epoch_val_rmses)
                fold_train_maes.append(epoch_train_maes)
                fold_val_maes.append(epoch_val_maes)
                fold_train_r2s.append(epoch_train_r2s)
                fold_val_r2s.append(epoch_val_r2s)
                fold_val_pearsons.append(epoch_val_pearsons)
                fold_val_spearmans.append(epoch_val_spearmans)
            print(f'Fold {fold+1} time: {time.time() - start_fold_time:.2f}s')
        self.labels = original_labels
        avg_val_loss = torch.tensor(fold_val_losses).mean(dim=0).tolist()
        avg_train_loss = torch.tensor(fold_train_losses).mean(dim=0).tolist()
        avg_epoch_time = torch.tensor(fold_epoch_times).mean(dim=0).tolist()
        last_n = 10
        avg_val_loss_last_n = torch.tensor(fold_val_losses)[:, -last_n:].mean().item()
        avg_train_loss_last_n = torch.tensor(fold_train_losses)[:, -last_n:].mean().item()
        avg_epoch_time_last_n = torch.tensor(fold_epoch_times)[:, -last_n:].mean().item()
        print(f'Average Train Loss (Last {last_n}): {avg_train_loss_last_n:.4f}')
        print(f'Average Val Loss (Last {last_n}): {avg_val_loss_last_n:.4f}')
        print(f'Average Epoch Time: {avg_epoch_time_last_n:.2f}s')
        if self.task == "Classification":
            avg_train_acc = torch.tensor(fold_train_accuracies).mean(dim=0).tolist()
            avg_val_acc = torch.tensor(fold_val_accuracies).mean(dim=0).tolist()
            avg_train_f1 = torch.tensor(fold_train_f1s).mean(dim=0).tolist()
            avg_val_f1 = torch.tensor(fold_val_f1s).mean(dim=0).tolist()
            avg_train_acc_last_n = torch.tensor(fold_train_accuracies)[:, -last_n:].mean().item()
            avg_val_acc_last_n = torch.tensor(fold_val_accuracies)[:, -last_n:].mean().item()
            avg_train_f1_last_n = torch.tensor(fold_train_f1s)[:, -last_n:].mean().item()
            avg_val_f1_last_n = torch.tensor(fold_val_f1s)[:, -last_n:].mean().item()
            print(f'Average Train Accuracy (Last {last_n}): {avg_train_acc_last_n:.4f}')
            print(f'Average Val Accuracy (Last {last_n}): {avg_val_acc_last_n:.4f}')
            print(f'Average Train F1 (Last {last_n}): {avg_train_f1_last_n:.4f}')
            print(f'Average Val F1 (Last {last_n}): {avg_val_f1_last_n:.4f}')
            self.plot_metrics(avg_train_loss, avg_val_loss, avg_train_acc, avg_val_acc, avg_train_f1, avg_val_f1, avg_epoch_time, self.input_dir)
            self.plot_roc_curve(fpr_list, tpr_list, auc_list, self.input_dir)
        elif self.task == "Regression":
            avg_train_rmse_last_n = torch.tensor(fold_train_rmses)[:, -last_n:].mean().item()
            avg_val_rmse_last_n = torch.tensor(fold_val_rmses)[:, -last_n:].mean().item()
            avg_train_mae_last_n = torch.tensor(fold_train_maes)[:, -last_n:].mean().item()
            avg_val_mae_last_n = torch.tensor(fold_val_maes)[:, -last_n:].mean().item()
            avg_train_r2_last_n = torch.tensor(fold_train_r2s)[:, -last_n:].mean().item()
            avg_val_r2_last_n = torch.tensor(fold_val_r2s)[:, -last_n:].mean().item()
            avg_val_pearson_last_n = torch.tensor(fold_val_pearsons)[:, -last_n:].mean().item()
            avg_val_spearman_last_n = torch.tensor(fold_val_spearmans)[:, -last_n:].mean().item()
            print(f'Average Train RMSE (Last {last_n}): {avg_train_rmse_last_n:.4f}')
            print(f'Average Val RMSE (Last {last_n}): {avg_val_rmse_last_n:.4f}')
            print(f'Average Train MAE (Last {last_n}): {avg_train_mae_last_n:.4f}')
            print(f'Average Val MAE (Last {last_n}): {avg_val_mae_last_n:.4f}')
            print(f'Average Train R2 (Last {last_n}): {avg_train_r2_last_n:.4f}')
            print(f'Average Val R2 (Last {last_n}): {avg_val_r2_last_n:.4f}')
            print(f'Average Val Pearson (Last {last_n}): {avg_val_pearson_last_n:.4f}')
            print(f'Average Val Spearman (Last {last_n}): {avg_val_spearman_last_n:.4f}')
            self.plot_regression_metrics(
                torch.tensor(fold_train_losses).mean(dim=0).tolist(),
                torch.tensor(fold_val_losses).mean(dim=0).tolist(),
                torch.tensor(fold_train_rmses).mean(dim=0).tolist(),
                torch.tensor(fold_val_rmses).mean(dim=0).tolist(),
                torch.tensor(fold_train_maes).mean(dim=0).tolist(),
                torch.tensor(fold_val_maes).mean(dim=0).tolist(),
                torch.tensor(fold_train_r2s).mean(dim=0).tolist(),
                torch.tensor(fold_val_r2s).mean(dim=0).tolist(),
                avg_epoch_time,
                self.input_dir,
            )

    def plot_regression_metrics(
        self,
        train_losses,
        val_losses,
        train_rmses,
        val_rmses,
        train_maes,
        val_maes,
        train_r2s,
        val_r2s,
        epoch_times,
        input_dir,
    ):
        os.makedirs(input_dir, exist_ok=True)
        epochs = range(1, len(train_losses) + 1)
        plt.figure(figsize=(18, 12))
        plt.subplot(2, 2, 1)
        plt.plot(epochs, train_losses, label='Training loss')
        plt.plot(epochs, val_losses, label='Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.subplot(2, 2, 2)
        plt.plot(epochs, train_rmses, label='Training RMSE')
        plt.plot(epochs, val_rmses, label='Validation RMSE')
        plt.xlabel('Epochs')
        plt.ylabel('RMSE')
        plt.legend()
        plt.title('Training and Validation RMSE')
        plt.subplot(2, 2, 3)
        plt.plot(epochs, train_maes, label='Training MAE')
        plt.plot(epochs, val_maes, label='Validation MAE')
        plt.xlabel('Epochs')
        plt.ylabel('MAE')
        plt.legend()
        plt.title('Training and Validation MAE')
        plt.subplot(2, 2, 4)
        plt.plot(epochs, train_r2s, label='Training R2')
        plt.plot(epochs, val_r2s, label='Validation R2')
        plt.plot(epochs, epoch_times, label='Epoch Time (s)', alpha=0.5)
        plt.xlabel('Epochs')
        plt.legend()
        plt.title('R2 and Epoch Time')
        plt.tight_layout()
        plot_path = os.path.join(input_dir, "cross_v_regression_metrics.png")
        plt.savefig(plot_path)
        print(f"Regression metrics plot saved to {plot_path}")

    def predict(
        self,
        smiles_list,
        compound_names,
        output_csv=None,
        regression_target_mean=0.0,
        regression_target_std=1.0,
    ):
        self.model.eval()
        results = []
        test_dataset = MolecularDataset(
            smiles_list,
            compound_names,
            labels=None,
            **self.get_preprocessing_kwargs()
        )
        test_loader = GeometricDataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)    
        for data in test_loader: 
            if data is None:
                print("Warning: Found None in data loader.")
                continue
            data = data.to(self.device)
            with torch.no_grad():
                out, _ = self.model(data.x, data.edge_index, data.edge_attr, data.u, data.batch)
            data_list = data.to_data_list()

            if self.task == "Classification":
                logits = out.view(-1)
                probs = torch.sigmoid(logits)
                pred = (probs > self.threshold).long()
                for mol_data, logit, prob, pred_label in zip(
                    data_list,
                    logits.detach().cpu().numpy().flatten(),
                    probs.detach().cpu().numpy().flatten(),
                    pred.detach().cpu().numpy().flatten(),
                ):
                    row = {
                        "SMILES": mol_data.smiles,
                        "Compound Name": mol_data.name,
                        "Raw Logit": float(logit),
                        "Probability": float(prob),
                        "Probability Threshold": float(self.threshold),
                        "Predicted Label": int(pred_label),
                    }
                    results.append(row)
                    logger.info(
                        f"Prediction: {mol_data.name}, Raw Logit: {logit:.8f}, "
                        f"Probability: {prob:.8f}, "
                        f"Probability Threshold: {self.threshold:.4f}, "
                        f"Predicted Label: {int(pred_label)}"
                    )
            elif self.task == "Regression":
                raw_predictions = out.view(-1).detach().cpu().numpy().flatten()
                predictions = raw_predictions * regression_target_std + regression_target_mean
                for mol_data, raw_prediction, prediction in zip(data_list, raw_predictions, predictions):
                    row = {
                        "SMILES": mol_data.smiles,
                        "Compound Name": mol_data.name,
                        "Raw Prediction": float(raw_prediction),
                        "Predicted Value": float(prediction),
                    }
                    results.append(row)
                    logger.info(
                        f"Prediction: {mol_data.name}, Raw Prediction: {raw_prediction:.8f}, "
                        f"Predicted Value: {prediction:.8f}"
                    )
            else:
                raise ValueError(f"Unsupported task type: {self.task}")
        if self.task == "Classification" and results:
            probabilities = np.asarray([row["Probability"] for row in results], dtype=float)
            positive_predictions = sum(row["Predicted Label"] for row in results)
            logger.info(
                f"Prediction summary: probability_min={probabilities.min():.6f}, "
                f"probability_max={probabilities.max():.6f}, "
                f"probability_mean={probabilities.mean():.6f}, "
                f"threshold={self.threshold:.6f}, "
                f"positive_predictions={positive_predictions}/{len(results)}"
            )
        elif self.task == "Regression" and results:
            predicted_values = np.asarray([row["Predicted Value"] for row in results], dtype=float)
            logger.info(
                f"Prediction summary: mean={predicted_values.mean():.6f}, "
                f"std={predicted_values.std():.6f}, "
                f"min={predicted_values.min():.6f}, max={predicted_values.max():.6f}"
            )
        if output_csv:
            output_dir = os.path.dirname(output_csv)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(output_csv, mode='wt', newline='') as csv_file:
                if self.task == "Classification":
                    fieldnames = [
                        "SMILES",
                        "Compound Name",
                        "Raw Logit",
                        "Probability",
                        "Probability Threshold",
                        "Predicted Label",
                    ]
                else:
                    fieldnames = [
                        "SMILES",
                        "Compound Name",
                        "Raw Prediction",
                        "Predicted Value",
                    ]
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
        return results

    def plot_metrics(self, train_losses, val_losses, train_accuracies, val_accuracies, train_f1s, val_f1s, epoch_times, input_dir):
        os.makedirs(input_dir, exist_ok=True)
        epochs = range(1, len(train_losses) + 1)
        plt.figure(figsize=(18, 12))
        plt.subplot(2, 2, 1)
        plt.plot(epochs, train_losses, label='Training loss')
        plt.plot(epochs, val_losses, label='Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.subplot(2, 2, 2)
        plt.plot(epochs, train_accuracies, label='Training accuracy')
        plt.plot(epochs, val_accuracies, label='Validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')
        plt.subplot(2, 2, 3)
        plt.plot(epochs, train_f1s, label='Training F1 Score')
        plt.plot(epochs, val_f1s, label='Validation F1 Score')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.title('Training and Validation F1 Score')
        plt.subplot(2, 2, 4)
        plt.plot(epochs, epoch_times, label='Epoch Time')
        plt.xlabel('Epochs')
        plt.ylabel('Time (s)')
        plt.legend()
        plt.title('Epoch Processing Time')
        plt.tight_layout()
        plot_path = os.path.join(input_dir, "cross_v_metrics.png")
        plt.savefig(plot_path)
        print(f"Metrics plot saved to {plot_path}")
    def plot_roc_curve(self, fpr_list, tpr_list, auc_list, input_dir):
        if not fpr_list:
            logger.warning("Skipping ROC plot because no validation fold had both classes.")
            return
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []
        for i in range(len(fpr_list)):
            interp_tpr = np.interp(mean_fpr, fpr_list[i], tpr_list[i])
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(auc_list[i])        
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)        
        plt.figure()
        plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})', lw=2)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plot_path = os.path.join(input_dir, "roc_auc.png")
        plt.savefig(plot_path)
        print(f"ROC Curve plot saved to {plot_path}")
