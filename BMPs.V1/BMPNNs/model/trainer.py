from .interaction_network import InteractionNetwork
import os
import csv
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch.utils.data import WeightedRandomSampler
from BMPNNs.data.molecular_dataset import MolecularDataset
import time 
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, root_mean_squared_error
)
from sklearn.model_selection import KFold
from matplotlib.colors import LinearSegmentedColormap
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from collections import defaultdict
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import random
input_dir="output"
os.makedirs(input_dir, exist_ok=True)
seed=122
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")
class GNNTrainer:
    def __init__(self, smiles_list, labels=None, names_list=None, node_block="BMP", hidden_channels=64, task="Classification", num_node_features=5, global_dim=6, lr=0.001, edge_dim=4, batch_size=32, k_folds=5, dropout_rate=0.5, input_dir = "evaluate_outputs", max_norm = 1, epochs = 50, threshold = 0.5):
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
        self.max_norm = max_norm
    def setup_model(self):
        self.model = InteractionNetwork(self.num_node_features, self.edge_dim, self.hidden_channels, self.global_dim, self.dropout_rate, self.node_block).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if self.task == "Regression":
            self.criterion = nn.MSELoss()
        elif self.task == "Classification":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported task type: {self.task}. Must be 'Classification' or 'Regression'.")
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5)
        print(f"\nMode: {self.node_block.upper() if isinstance(self.node_block, str) else type(self.node_block).__name__}")
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
        all_labels = []
        all_names = []  
        all_predictions = [] 
        for data in train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            data.x = data.x.to(device)
            data.edge_index = data.edge_index.to(device)
            data.edge_attr = data.edge_attr.to(device)
            data.u = data.u.to(device)
            data.batch = data.batch.to(device)
            data.y = data.y.view(-1, 1)  

            out, _ = self.model(
                data.x, data.edge_index, data.edge_attr, data.u, data.batch
            )
            loss = self.criterion(out, data.y)
            loss.backward() 
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)
            self.optimizer.step()
            total_loss += loss.item() * data.num_graphs
            probabilities = torch.sigmoid(out) 
            pred = (probabilities > self.threshold).float()  
            all_labels.append(data.y.cpu().numpy())
            all_names.extend(data.name)  
            all_predictions.extend(pred.detach().cpu().numpy().flatten()) 
        self.all_labels = np.concatenate(all_labels) 
        self.all_names = all_names 
        self.all_predictions = np.array(all_predictions) 
        if len(self.all_labels) != len(self.all_names) or len(self.all_labels) != len(self.all_predictions):
            raise ValueError("Mismatch: all_labels, all_names, and all_predictions lengths are not equal.")
        return total_loss / len(train_loader.dataset) 
    def save_molecule_image_with_importance(self, mol, name, atom_importance, cmap, output_dir=None, img_size=(400, 400)):
        if output_dir is None:
            output_dir = self.input_dir
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
    def evaluate(self, loader, generate_images=True):
        self.model.eval()
        molecule_cache = {}
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
        all_raw_node_scores = []
        compound_names = []  
        correct = 0  
        for data in loader: 
            data = data.to(self.device)
            with torch.no_grad():
                out, atom_importance = self.model(data.x, data.edge_index, data.edge_attr, data.u, data.batch)
                loss = self.criterion(out.view(-1), data.y.view(-1).float())
                total_loss += loss.item() * data.num_graphs
            data_list = data.to_data_list()
            atom_importance_list = atom_importance.detach().cpu().split([d.num_nodes for d in data_list])
            for mol_data, mol_importance in zip(data_list, atom_importance_list):
                mol_name = mol_data.name
                mol = Chem.MolFromSmiles(mol_data.smiles)
                min_val = mol_importance.min().item()
                max_val = mol_importance.max().item()
                if max_val > min_val:
                    normalized_importance = (mol_importance - min_val) / (max_val - min_val)
                else:
                    normalized_importance = torch.zeros_like(mol_importance)
                if generate_images:
                    self.save_molecule_image_with_importance(
                        mol, mol_name, normalized_importance, cmap
                    )
                    normalized_scores = normalized_importance.detach().cpu().numpy()
                    with open(f"{self.input_dir}/{mol_name}_node_scores.txt", "w") as f:
                        for i, score in enumerate(normalized_scores):
                            atom_symbol = mol.GetAtomWithIdx(i).GetSymbol()
                            f.write(f"Atom {i} ({atom_symbol}): {score:.4f}\n")
            all_targets.extend(data.y.cpu().numpy().flatten())
            all_preds.extend(out.detach().cpu().numpy().flatten())
            compound_names.extend([d.name for d in data_list]) 
            if self.task == "Classification":
                pred = (out > self.threshold).float()
                correct += (pred == data.y).sum().item()
        if self.task == "Classification":
            all_preds_binarized = [1 if p > self.threshold else 0 for p in all_preds]
            accuracy = accuracy_score(all_targets, all_preds_binarized)
            precision = precision_score(all_targets, all_preds_binarized)
            recall = recall_score(all_targets, all_preds_binarized)
            f1 = f1_score(all_targets, all_preds_binarized)
            return accuracy, precision, recall, f1, total_loss / len(loader.dataset), all_targets, all_preds, compound_names, all_preds_binarized
        elif self.task == "Regression":
            rmse = root_mean_squared_error(all_targets, all_preds)
            return rmse, total_loss / len(loader.dataset), all_targets, all_preds, compound_names
        else:
            raise ValueError(f"Unsupported task type: {self.task}")

    def cross_validate(self):
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=seed)
        fold_val_losses = []
        fold_train_losses = []
        fold_epoch_times = []
        if self.task == "Classification":
            fold_train_accuracies, fold_val_accuracies = [], []
            fold_train_f1s, fold_val_f1s = [], []
            fpr_list, tpr_list, auc_list = [], [], []
        elif self.task == "Regression":
            fold_train_rmses, fold_val_rmses = [], []
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.smiles_list)):
            start_fold_time = time.time()
            print(f'Fold {fold + 1}/{self.k_folds}')
            smiles_train = [self.smiles_list[i] for i in train_idx]
            names_train = [self.names_list[i] for i in train_idx]
            labels_train = [self.labels[i] for i in train_idx]
            smiles_val = [self.smiles_list[i] for i in val_idx]
            labels_val = [self.labels[i] for i in val_idx]
            names_val = [self.names_list[i] for i in val_idx]
            train_dataset = MolecularDataset(smiles_train, names_train, labels_train)
            val_dataset = MolecularDataset(smiles_val, names_val, labels_val)
            self.global_dim = train_dataset.global_dim
            self.edge_dim = train_dataset.edge_dim
            self.num_node_features = train_dataset.num_node_features  
            train_loader = GeometricDataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
            val_loader = GeometricDataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)    
            self.setup_model()
            epoch_train_losses, epoch_val_losses = [], []
            epoch_times = []
            if self.task == "Classification":
                epoch_train_accuracies, epoch_val_accuracies = [], []
                epoch_train_f1s, epoch_val_f1s = [], []
            elif self.task == "Regression":
                epoch_train_rmses, epoch_val_rmses = [], []
            for epoch in range(self.epochs):
                start_time = time.time()
                _ = self.train(train_loader)
                if self.task == "Classification":
                    train_acc, _, _, train_f1, train_loss, _, _, _, _ = self.evaluate(train_loader, generate_images=False)
                    val_acc, _, _, val_f1, val_loss, targets, preds, _, _ = self.evaluate(val_loader, generate_images=False)
                elif self.task == "Regression":
                    train_rmse, train_loss, _, _, _ = self.evaluate(train_loader, generate_images=False)
                    val_rmse, val_loss, targets, preds, _ = self.evaluate(val_loader, generate_images=False)
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
                    fpr, tpr, _ = roc_curve(targets, preds)
                    roc_auc = auc(fpr, tpr)
                    fpr_list.append(fpr)
                    tpr_list.append(tpr)
                    auc_list.append(roc_auc)
                    print(f'Epoch {epoch+1} | Loss: {train_loss:.4f}/{val_loss:.4f} | '
                        f'Acc: {train_acc:.4f}/{val_acc:.4f} | '
                        f'F1: {train_f1:.4f}/{val_f1:.4f} | Time: {epoch_time:.2f}s | LR: {current_lr:.6f}')
                elif self.task == "Regression":
                    epoch_train_rmses.append(train_rmse)
                    epoch_val_rmses.append(val_rmse)
                    print(f'Epoch {epoch+1} | Loss: {train_loss:.4f}/{val_loss:.4f} | '
                        f'RMSE: {train_rmse:.4f}/{val_rmse:.4f} | Time: {epoch_time:.2f}s | LR: {current_lr:.6f}')
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
            print(f'Fold {fold+1} time: {time.time() - start_fold_time:.2f}s')
        avg_val_loss = torch.tensor(fold_val_losses).mean(dim=0).tolist()
        avg_train_loss = torch.tensor(fold_train_losses).mean(dim=0).tolist()
        avg_epoch_time = torch.tensor(fold_epoch_times).mean(dim=0).tolist()
        last_n = 10
        avg_val_loss_last_n = torch.tensor(fold_val_losses)[:, -last_n:].mean(dim=1).tolist()
        avg_train_loss_last_n = torch.tensor(fold_train_losses)[:, -last_n:].mean(dim=1).tolist()
        avg_epoch_time_last_n = torch.tensor(fold_epoch_times)[:, -last_n:].mean(dim=1).tolist()
        print(f'Average Train Loss (Last {last_n}): {avg_train_loss_last_n[-1]:.4f}')
        print(f'Average Val Loss (Last {last_n}): {avg_val_loss_last_n[-1]:.4f}')
        print(f'Average Epoch Time: {avg_epoch_time_last_n[-1]:.2f}s')
        if self.task == "Classification":
            avg_train_acc = torch.tensor(fold_train_accuracies).mean(dim=0).tolist()
            avg_val_acc = torch.tensor(fold_val_accuracies).mean(dim=0).tolist()
            avg_train_f1 = torch.tensor(fold_train_f1s).mean(dim=0).tolist()
            avg_val_f1 = torch.tensor(fold_val_f1s).mean(dim=0).tolist()
            avg_train_acc_last_n = torch.tensor(fold_train_accuracies)[:, -last_n:].mean(dim=1).tolist()
            avg_val_acc_last_n = torch.tensor(fold_val_accuracies)[:, -last_n:].mean(dim=1).tolist()
            avg_train_f1_last_n = torch.tensor(fold_train_f1s)[:, -last_n:].mean(dim=1).tolist()
            avg_val_f1_last_n = torch.tensor(fold_val_f1s)[:, -last_n:].mean(dim=1).tolist()
            print(f'Average Train Accuracy (Last {last_n}): {avg_train_acc_last_n[-1]:.4f}')
            print(f'Average Val Accuracy (Last {last_n}): {avg_val_acc_last_n[-1]:.4f}')
            print(f'Average Train F1 (Last {last_n}): {avg_train_f1_last_n[-1]:.4f}')
            print(f'Average Val F1 (Last {last_n}): {avg_val_f1_last_n[-1]:.4f}')
            self.plot_metrics(avg_train_loss, avg_val_loss, avg_train_acc, avg_val_acc, avg_train_f1, avg_val_f1, avg_epoch_time, self.input_dir)
            self.plot_roc_curve(fpr_list, tpr_list, auc_list, self.input_dir)
        elif self.task == "Regression":
            #avg_train_rmse = torch.tensor(fold_train_rmses).mean(dim=0).tolist()
            #avg_val_rmse = torch.tensor(fold_val_rmses).mean(dim=0).tolist()
            avg_train_rmse_last_n = torch.tensor(fold_train_rmses)[:, -last_n:].mean(dim=1).tolist()
            avg_val_rmse_last_n = torch.tensor(fold_val_rmses)[:, -last_n:].mean(dim=1).tolist()
            print(f'Average Train RMSE (Last {last_n}): {avg_train_rmse_last_n[-1]:.4f}')
            print(f'Average Val RMSE (Last {last_n}): {avg_val_rmse_last_n[-1]:.4f}')
    def predict(self, smiles_list, compound_names, output_csv=None):
        self.model.eval()
        predictions = []
        test_dataset = MolecularDataset(smiles_list, compound_names, labels=None)
        test_loader = GeometricDataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)    
        smiles_strings = []    
        predicted_names = [] 
        for data in test_loader: 
            if data is None:
                print("Warning: Found None in data loader.")
                continue
            data = data.to(self.device)
            out, _ = self.model(data.x, data.edge_index, data.edge_attr, data.u, data.batch)

            if self.task == "Classification":
                pred = (out > self.threshold).float()
                pred = pred.detach().cpu().numpy().flatten()
            elif self.task == "Regression":
                pred = out.detach().cpu().numpy().flatten()
            else:
                raise ValueError(f"Unsupported task type: {self.task}")
            predictions.extend(pred)
            smiles_strings.extend([d.smiles for d in data.to_data_list()])
            predicted_names.extend([d.name for d in data.to_data_list()])
        results = list(zip(smiles_strings, predicted_names, predictions))  
        if output_csv:
            with open(output_csv, mode='wt', newline='') as csv_file:
                writer = csv.writer(csv_file)
                header = ['SMILES', 'Compound Name', 'Predicted Value' if self.task == 'Regression' else 'Predicted Label']
                writer.writerow(header)
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
