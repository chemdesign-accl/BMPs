from rdkit import Chem
import numpy as np
from mendeleev import element
from molvs import Standardizer
import torch
from torch.utils.data import TensorDataset, WeightedRandomSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch_geometric.utils import softmax
import torch_geometric.transforms as T
import logging
import os
logger = logging.getLogger(__name__)
from rdkit.Chem import (
    AllChem, Draw, Descriptors
)
class MolecularDataset:
    def __init__(self, smiles_list, names_list, labels=None, node_block="BMP"):
        self.smiles_list = smiles_list.copy()
        self.names_list = names_list.copy()
        self.labels = labels.copy() if labels is not None else [None] * len(smiles_list)
        self.data_list = []
        self.node_block = node_block
        self.global_dim = 0
        self.num_node_features = 0
        self.edge_dim = 0
        self.successful_labels = []
        self._mendeleev_cache = {}
        self.successful_names = [] 
        self.successful_smiles = [] 
        self.hybridization_dict = {
            Chem.rdchem.HybridizationType.SP: 0,
            Chem.rdchem.HybridizationType.SP2: 0.5,
            Chem.rdchem.HybridizationType.SP3: 1,
        }
        self.processed_count = 0
        print(f"Number of molecules in dataset: {len(smiles_list)}")
        print(f"Using Model: {self.node_block}")

        logger.info("Converting SMILES to data objects.")
        i = 0
        while i < len(self.smiles_list):
            smiles = self.smiles_list[i]
            name = self.names_list[i]
            label = self.labels[i]
            try:
                data = self.smiles_to_data(smiles, name, label)
                if data is not None:
                    self.data_list.append(data)
                    self.successful_labels.append(label)
                    self.successful_names.append(name)
                    self.successful_smiles.append(smiles)

                    self.processed_count += 1
                else:
                    logger.warning(f"Skipping invalid molecule: Name: {name}, SMILES: {smiles}")
                    del self.smiles_list[i]
                    del self.names_list[i]
                    del self.labels[i]
                    continue
            except Exception as e:
                logger.error(f"Failed to process molecule: Name: {name}, SMILES: {smiles}, Error: {e}")
                del self.smiles_list[i]
                del self.names_list[i]
                del self.labels[i]
                continue
            i += 1
        logger.info(f"Processed {self.processed_count} valid molecules out of {len(smiles_list)} provided.")
    def smiles_to_data(self, smiles, name, label=None, output_dir="molecule_images"):
        s = Standardizer()  
        try:
            mol = Chem.MolFromSmiles(smiles)
            mol = s.standardize(mol)
            mol = self.correct_atom_types(mol)
            if mol is None:
                logger.warning(f"Failed to parse SMILES: {smiles}")
                return None
            try:
                mol_with_h = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol_with_h, enforceChirality=True)
                AllChem.MMFFOptimizeMolecule(mol_with_h)
                Chem.AssignStereochemistryFrom3D(mol_with_h, replaceExistingTags=True)
                conf_with_h = mol_with_h.GetConformer()
                mol = Chem.RemoveHs(mol_with_h)
                conf = Chem.Conformer(mol.GetNumAtoms())
                for atom_id in range(mol.GetNumAtoms()):
                    pos = conf_with_h.GetAtomPosition(atom_id)
                    conf.SetAtomPosition(atom_id, pos)
                mol.AddConformer(conf)  
            except Exception as e:
                logger.error(f"Embedding, optimization, or sanitization failed: {e}, SMILES: {smiles}")
                return None
            return self.extract_features(mol, conf, name, smiles, label, output_dir)
        except Exception as e:
            logger.error(f"General failure processing SMILES: {smiles}, Error: {e}")
            return None 
    def extract_features(self, mol, conf, name, smiles, label, output_dir):
        try:
            atom_features = self.get_atom_features(mol, conf)
            edge_index, edge_attr = self.get_edge_index_and_features(mol, conf, self.node_block)
            if edge_index.max().item() >= atom_features.size(0):
                logger.error(f"Invalid edge index detected: {edge_index.max().item()} exceeds number of atoms {atom_features.size(0)}")
                return None
            global_features = self.get_global_features(mol, conf)
            if label is not None:
                target = torch.tensor([label], dtype=torch.float).reshape(-1, 1)
                data = Data(x=atom_features, edge_index=edge_index, edge_attr=edge_attr, u=global_features, y=target,)
            else:
                data = Data(x=atom_features, edge_index=edge_index, edge_attr=edge_attr, u=global_features)        
            data.smiles = smiles
            data.name = name
            #self.save_molecule_image(mol, name, output_dir)
            logger.info(f"Processed {self.processed_count} molecules so far.")
            return data
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}, SMILES: {smiles}")
            return None
    def min_max_normalize(self, value, min_value, max_value):
        if max_value == min_value:
            logger.warning(f"Normalization skipped: min_value == max_value for {value}")
            return 0
        normalized = (value - min_value) / (max_value - min_value)
        return normalized
    def get_cached_element_props(self, atomic_num):
        if atomic_num not in self._mendeleev_cache:
            el = element(atomic_num)
            self._mendeleev_cache[atomic_num] = {
                "electronegativity": (el.electronegativity('pauling') - 0.9) / 3.1,
                "polarizability": (el.dipole_polarizability - 4.5) / (35 - 4.5),
                "vdw_radius": (el.vdw_radius - 120) / (166 - 120)
            }
        return self._mendeleev_cache[atomic_num]
    def calculate_radius_of_gyration(self, mol, conf):
        try:
            coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
            masses = np.array([atom.GetMass() for atom in mol.GetAtoms()])
            total_mass = np.sum(masses)
            center_of_mass = np.sum(coords.T * masses, axis=1) / total_mass
            rg_square = np.sum(masses * np.sum((coords - center_of_mass) ** 2, axis=1)) / total_mass
            radius_of_gyration = np.sqrt(rg_square)
            return radius_of_gyration
        except Exception as e:
            logger.error(f"Radius of gyration failed: {e}")
    def get_global_features(self, mol, conf):
        global_features = [                        
            len(Chem.FindMolChiralCenters(mol, includeUnassigned=False))/6, 
            abs(1/(10 * (Descriptors.NumHDonors(mol) / 5) + abs(Descriptors.NumHAcceptors(mol) / 10))),
            Descriptors.NumRotatableBonds(mol)/10,
            (Descriptors.TPSA(mol) + Descriptors.MolLogP(mol))/145,
            Descriptors.FractionCSP3(mol),
            self.calculate_radius_of_gyration(mol, conf)/5,
            ]
        self.global_dim = len(global_features)
        return torch.tensor(global_features, dtype=torch.float).unsqueeze(0)
    def correct_atom_types(self, mol):
        corrections = {
            "Cu+2": 29,   # Copper (Cu+2)
            "Se+2": 34,   # Selenium (Se+2)
            "Rh+6": 45,   # Rhodium (Rh+6)
            "W+6": 74,    # Tungsten (W+6)
            "Co+3": 27,   # Cobalt (Co+3)
            "Zn+2": 30,   # Zinc (Zn+2)
            "Ni+2": 28,   # Nickel (Ni+2)
            "Pd+2": 46,   # Palladium (Pd+2)
            "Gd+3": 64,   # Gadolinium (Gd+3)
            "Re+5": 75,   # Rhenium (Re+5)
            "Pt+2": 78,   # Platinum (Pt+2)
            "Cr3+3": 24,  # Chromium (Cr3+3)
            "Zr2": 40,    # Zirconium (Zr2)
            "Ba": 56,     # Barium (Ba0)
            "Ba1": 56,    # Barium (Ba1)
            "Pd6+2": 46,  # Palladium (Pd6+2)
            "Cr2+3": 24,  # Chromium (Cr2+3)
            "Cr1+3": 24,  # Chromium (Cr1+3)
            "Fe2+2": 26,  # Iron (Fe2+2)
            "Au+3": 79,   # Gold (Au+3)
            "Ca+2": 20,   # Calcium (Ca+2)
            "Cu5+1": 29,  # Copper (Cu5+1)
            "Cr+3": 24,   # Chromium (Cr+3)
            "Zr": 40,     # Zirconium (Zr)
            "Pd3+2": 46,  # Palladium (Pd3+2)
            "Co3+3": 27,  # Cobalt (Co3+3)
            "Pb3+3": 82,  # Lead (Pb3+3)
            "In2+3": 49,  # Indium (In2+3)
            "Pt2+2": 78,  # Platinum (Pt2+2)
            "Se2+2": 34,  # Selenium (Se2+2)
            "Mn2+2": 25,  # Manganese (Mn2+2)
            "Be+2": 4,    # Beryllium (Be+2)
            "Au5+3": 79,  # Gold (Au5+3)
            "Fe1+2": 26,  # Iron (Fe1+2)
            "Ti+4": 22,   # Titanium (Ti+4)
        }
        for atom in mol.GetAtoms():
            formal_charge = atom.GetFormalCharge()
            symbol = atom.GetSymbol()
            charge_sign = "+" if formal_charge >= 0 else ""
            key = f"{symbol}{charge_sign}{formal_charge}"
            if key in corrections:
                atomic_num = corrections[key]
                print(f"Correcting {key} to atomic number {atomic_num}")
                atom.SetAtomicNum(atomic_num)
                atom.SetFormalCharge(formal_charge)
            else:
                continue
        return mol
    def calculate_buried_volume(self,mol,  conf, atom_idx, radius=3.5, grid_spacing=0.5):
        central_atom_pos = conf.GetAtomPosition(atom_idx)
        central_point = np.array([central_atom_pos.x, central_atom_pos.y, central_atom_pos.z])
        grid = np.arange(-radius, radius + grid_spacing, grid_spacing)
        grid_points = np.array(np.meshgrid(grid, grid, grid)).reshape(3, -1).T
        sphere_mask = np.linalg.norm(grid_points, axis=1) <= radius
        sphere_points = grid_points[sphere_mask] + central_point
        occupied_count = 0
        for i, atom in enumerate(mol.GetAtoms()):
            if i == atom_idx:
                continue  
            atom_pos = conf.GetAtomPosition(i)
            atom_pos_array = np.array([atom_pos.x, atom_pos.y, atom_pos.z])
            distance = np.linalg.norm(sphere_points - atom_pos_array, axis=1)
            vdw_radius = Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum())
            occupied_count += np.sum(distance <= vdw_radius)
        total_points = len(sphere_points)
        buried_volume = occupied_count / total_points 
        return buried_volume
    def save_molecule_image(self, mol, name, output_dir="molecule_images", img_size=(200, 200)):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        img = Draw.MolToImage(mol, size=img_size)
        img_path = os.path.join(output_dir, f"molecule_{name}.png")
        img.save(img_path)          
    def hybridization_to_index(self, hybridization):
        return self.hybridization_dict.get(hybridization, 0)
    def get_atom_features(self, mol, conf):
        atom_features = []
        for atom in mol.GetAtoms():                

            if atom.GetAtomicNum() == 1:  
                continue
            atomic_num = atom.GetAtomicNum()
            props = self.get_cached_element_props(atomic_num)
            atom_feature = [
                (atomic_num - 1)/178,
                self.calculate_buried_volume(mol, conf, atom.GetIdx()),
                self.hybridization_dict.get(atom.GetHybridization(), 0),
                props["electronegativity"],
                props["polarizability"],
                props["vdw_radius"]
            ]
            atom_features.append(atom_feature)
        self.num_node_features = len(atom_feature)
        return torch.tensor(atom_features, dtype=torch.float)
    def get_ring_size_feature(self, bond):
        if not bond.IsInRing():
            return 0.0
        elif bond.IsInRingSize(3):
            return 0.14
        elif bond.IsInRingSize(4):
            return 0.28
        elif bond.IsInRingSize(5):
            return 0.42
        elif bond.IsInRingSize(6):
            return 0.57
        elif bond.IsInRingSize(7):
            return 0.71
        elif bond.IsInRingSize(8):
            return 0.85
        else:
            return 1.0          
    def get_edge_index_and_features(self, mol, conf, node_block):
        edge_index = []
        edge_attr = []
        try:
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                if i >= mol.GetNumAtoms() or j >= mol.GetNumAtoms():
                    print(f"Invalid atom index: i={i}, j={j}, num_atoms={mol.GetNumAtoms()}")
                    continue
                bond_length = Chem.rdMolTransforms.GetBondLength(conf, i, j)
                edge_feature = [
                    (bond_length -1.05161541)/(2.4620574 - 1.05161541),
                    bond.GetBondTypeAsDouble()/2,
                    1 if bond.GetIsConjugated() else 0,  
                    self.get_ring_size_feature(bond)  
                ]
                if node_block == "UMP":
                    edge_index.append([i, j])
                    edge_index.append([j, i])
                    edge_attr.append(edge_feature)
                    edge_attr.append(edge_feature)  
                else:
                    edge_index.append([i, j])
                    edge_attr.append(edge_feature)

        except Exception as e:
            print(f"Error processing bond features for molecule: {e}")
            return None, None
        self.edge_dim = len(edge_feature)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)   
        return edge_index, edge_attr
    def get(self, idx):
        if idx >= len(self.data_list) or idx < 0:
            raise IndexError(f"Index {idx} is out of bounds for data_list of length {len(self.data_list)}")
        return self.data_list[idx]
    def indices(self):
        return range(self.len())
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        if not isinstance(idx, int):
            raise TypeError(f"Index must be an integer, but got {type(idx)}")
        if idx >= len(self.data_list) or idx < 0:
            raise IndexError(f"Index {idx} is out of bounds for data_list of length {len(self.data_list)}")
        return self.data_list[idx]