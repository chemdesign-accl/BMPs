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
import time
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
logger = logging.getLogger(__name__)
from rdkit.Chem import AllChem, Draw, Descriptors
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)
_MENDELEEV_CACHE = {}
_COMMON_ATOMIC_NUMS = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53}

def _clean_smiles(smiles):
    if smiles is None:
        return None
    if not isinstance(smiles, str):
        return None
    smiles = smiles.strip()
    if not smiles or smiles.lower() in {"nan", "none", "null"}:
        return None
    return smiles

def _get_element_props(atomic_num):
    if atomic_num not in _MENDELEEV_CACHE:
        el = element(atomic_num)
        _MENDELEEV_CACHE[atomic_num] = {
            "electronegativity": (el.electronegativity('pauling') - 0.9) / 3.1,
            "polarizability": (el.dipole_polarizability - 4.5) / (35 - 4.5),
            "vdw_radius": (el.vdw_radius - 120) / (166 - 120)
        }
    return _MENDELEEV_CACHE[atomic_num]

def _atomic_numbers_from_smiles(smiles_list):
    atomic_nums = set(_COMMON_ATOMIC_NUMS)
    for smiles in smiles_list:
        smiles = _clean_smiles(smiles)
        if smiles is None:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        atomic_nums.update(atom.GetAtomicNum() for atom in mol.GetAtoms())
    return atomic_nums

def _serialize_data_for_worker(data, include_y=True):
    payload = {
        "x": data.x.cpu().numpy(),
        "edge_index": data.edge_index.cpu().numpy(),
        "edge_attr": data.edge_attr.cpu().numpy(),
        "u": data.u.cpu().numpy(),
        "smiles": data.smiles,
        "name": data.name,
    }
    y = getattr(data, "y", None)
    if include_y and y is not None:
        payload["y"] = y.cpu().numpy()
    return payload

def _deserialize_worker_data(payload):
    kwargs = {
        "x": torch.tensor(payload["x"], dtype=torch.float),
        "edge_index": torch.tensor(payload["edge_index"], dtype=torch.long),
        "edge_attr": torch.tensor(payload["edge_attr"], dtype=torch.float),
        "u": torch.tensor(payload["u"], dtype=torch.float),
    }
    if "y" in payload:
        kwargs["y"] = torch.tensor(payload["y"], dtype=torch.float)
    data = Data(**kwargs)
    data.smiles = payload.get("smiles", "")
    data.name = payload.get("name", "")
    return data

def _process_molecule_worker(args):
    idx, smiles, name, label, node_block, options = args
    worker = MolecularDataset.__new__(MolecularDataset)
    worker.node_block = node_block
    worker.global_dim = 0
    worker.num_node_features = 0
    worker.edge_dim = 0
    worker.processed_count = 0
    worker.timing_totals = {}
    worker._mendeleev_cache = options["mendeleev_cache"].copy()
    worker.standardizer = Standardizer()
    worker.suppress_timing_logs = True
    worker.save_images = options["save_images"]
    worker.num_confs = options["num_confs"]
    worker.max_isomers = options["max_isomers"]
    worker.stereo_try_embedding = options["stereo_try_embedding"]
    worker.embed_num_threads = options["embed_num_threads"]
    worker.buried_volume_radius = options["buried_volume_radius"]
    worker.buried_volume_grid_spacing = options["buried_volume_grid_spacing"]
    worker.hybridization_dict = {
        Chem.rdchem.HybridizationType.SP: 0,
        Chem.rdchem.HybridizationType.SP2: 0.5,
        Chem.rdchem.HybridizationType.SP3: 1,
    }
    try:
        data = worker.smiles_to_data(smiles, name, label)
        if data is None:
            return {
                "idx": idx,
                "data": None,
                "smiles": smiles,
                "name": name,
                "label": label,
                "error": None,
            }
        return {
            "idx": idx,
            "data": _serialize_data_for_worker(data),
            "smiles": smiles,
            "name": name,
            "label": label,
            "global_dim": worker.global_dim,
            "num_node_features": worker.num_node_features,
            "edge_dim": worker.edge_dim,
            "timing_totals": worker.timing_totals,
            "error": None,
        }
    except Exception as e:
        return {
            "idx": idx,
            "data": None,
            "smiles": smiles,
            "name": name,
            "label": label,
            "error": str(e),
        }

class MolecularDataset:
    def __init__(
        self,
        smiles_list,
        names_list,
        labels=None,
        node_block="BMP",
        num_workers=1,
        save_images=False,
        num_confs=2,
        max_isomers=8,
        stereo_try_embedding=False,
        embed_num_threads=1,
        buried_volume_radius=3.5,
        buried_volume_grid_spacing=0.5,
        cache_dir=None,
        use_cache=True,
    ):
        self.smiles_list = list(smiles_list)
        self.names_list = list(names_list)
        self.labels = list(labels) if labels is not None else [None] * len(self.smiles_list)
        if len(self.names_list) != len(self.smiles_list):
            raise ValueError(
                "names_list length must match smiles_list length: "
                f"{len(self.names_list)} != {len(self.smiles_list)}"
            )
        if len(self.labels) != len(self.smiles_list):
            raise ValueError(
                "labels length must match smiles_list length: "
                f"{len(self.labels)} != {len(self.smiles_list)}"
            )
        self.data_list = []
        self.node_block = node_block
        self.global_dim = 0
        self.num_node_features = 0
        self.edge_dim = 0
        self.successful_labels = []
        self._mendeleev_cache = _MENDELEEV_CACHE
        self.successful_names = [] 
        self.successful_smiles = [] 
        self.standardizer = Standardizer()
        self.suppress_timing_logs = False
        self.save_images = save_images
        self.num_confs = num_confs
        self.max_isomers = max_isomers
        self.stereo_try_embedding = stereo_try_embedding
        self.embed_num_threads = embed_num_threads
        self.buried_volume_radius = buried_volume_radius
        self.buried_volume_grid_spacing = buried_volume_grid_spacing
        self.cache_dir = cache_dir
        self.use_cache = use_cache and cache_dir is not None
        if self.use_cache:
            logger.info(f"Processed molecule cache enabled: {self.cache_dir}")
            logger.info(f"Processed molecule cache settings: {self.preprocessing_cache_settings()}")
        else:
            logger.info("Processed molecule cache disabled because cache_dir was not provided.")
        self.hybridization_dict = {
            Chem.rdchem.HybridizationType.SP: 0,
            Chem.rdchem.HybridizationType.SP2: 0.5,
            Chem.rdchem.HybridizationType.SP3: 1,
        }
        self.processed_count = 0
        self.timing_totals = {}
        self.num_workers = num_workers
        print(f"Number of molecules in dataset: {len(smiles_list)}")
        print(f"Using Model: {self.node_block}")
        logger.info("Converting SMILES to data objects.")
        self.prewarm_mendeleev_cache()
        if self.num_workers and self.num_workers > 1:
            self.process_molecules_parallel()
        else:
            self.process_molecules_sequential()
        logger.info(f"Processed {self.processed_count} valid molecules out of {len(smiles_list)} provided.")
        self.log_aggregate_timing_summary()
    def process_molecules_sequential(self):
        i = 0
        while i < len(self.smiles_list):
            smiles = self.smiles_list[i]
            name = self.names_list[i]
            label = self.labels[i]
            clean_smiles = _clean_smiles(smiles)
            if clean_smiles is None:
                logger.warning(
                    f"Skipping invalid molecule with missing/non-string SMILES: "
                    f"Name: {name}, SMILES: {smiles}"
                )
                del self.smiles_list[i]
                del self.names_list[i]
                del self.labels[i]
                continue
            smiles = clean_smiles
            self.smiles_list[i] = smiles
            try:
                data = self.load_cached_molecule(smiles, name, label)
                if data is not None:
                    logger.info(f"Loaded cached molecule: Name: {name}, SMILES: {smiles}")
                else:
                    data = self.smiles_to_data(smiles, name, label)
                    if data is not None:
                        self.save_cached_molecule(data, smiles)
                if data is not None:
                    self.data_list.append(data)
                    self.successful_labels.append(label)
                    self.successful_names.append(name)
                    self.successful_smiles.append(smiles)

                    self.processed_count += 1
                    logger.info(
                        f"Processed {self.processed_count}/{len(self.smiles_list)} molecules."
                    )
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
    def process_molecules_parallel(self):
        max_workers = min(self.num_workers, len(self.smiles_list))
        if max_workers < 1:
            return
        logger.info(f"Processing molecules with {max_workers} CPU workers.")
        options = {
            "save_images": self.save_images,
            "num_confs": self.num_confs,
            "max_isomers": self.max_isomers,
            "stereo_try_embedding": self.stereo_try_embedding,
            "embed_num_threads": self.embed_num_threads,
            "buried_volume_radius": self.buried_volume_radius,
            "buried_volume_grid_spacing": self.buried_volume_grid_spacing,
            "mendeleev_cache": self._mendeleev_cache,
        }
        results = []
        tasks = []
        for idx, (smiles, name, label) in enumerate(
            zip(self.smiles_list, self.names_list, self.labels)
        ):
            clean_smiles = _clean_smiles(smiles)
            if clean_smiles is None:
                logger.warning(
                    f"Skipping invalid molecule with missing/non-string SMILES: "
                    f"Name: {name}, SMILES: {smiles}"
                )
                results.append(
                    {
                        "idx": idx,
                        "data": None,
                        "smiles": smiles,
                        "name": name,
                        "label": label,
                        "error": None,
                    }
                )
                continue
            smiles = clean_smiles
            cached_data = self.load_cached_molecule(smiles, name, label)
            if cached_data is not None:
                results.append(
                    {
                        "idx": idx,
                        "data": _serialize_data_for_worker(cached_data),
                        "smiles": smiles,
                        "name": name,
                        "label": label,
                        "global_dim": cached_data.u.size(-1),
                        "num_node_features": cached_data.x.size(-1),
                        "edge_dim": cached_data.edge_attr.size(-1),
                        "timing_totals": {},
                        "error": None,
                    }
                )
            else:
                tasks.append((idx, smiles, name, label, self.node_block, options))
        if results:
            logger.info(f"Loaded {len(results)} molecules from processed cache.")
        if not tasks:
            logger.info("All molecules loaded from processed cache.")
            self.finalize_parallel_results(results)
            return
        task_indices = {task[0] for task in tasks}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(_process_molecule_worker, task): task for task in tasks
            }
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                except Exception as e:
                    _, smiles, name, _, _, _ = task
                    logger.error(
                        f"Failed to process molecule: Name: {name}, "
                        f"SMILES: {smiles}, Error: {e}"
                    )
                    continue
                results.append(result)
                if result["error"]:
                    logger.error(
                        f"Failed to process molecule: Name: {result['name']}, "
                        f"SMILES: {result['smiles']}, Error: {result['error']}"
                    )
                elif result["data"] is None:
                    logger.warning(
                        f"Skipping invalid molecule: Name: {result['name']}, "
                        f"SMILES: {result['smiles']}"
                    )
                else:
                    self.save_cached_payload(result["data"], result["smiles"])
                    completed = sum(
                        1 for item in results
                        if item["data"] is not None and item["idx"] in task_indices
                    )
                    if completed == 1 or completed % 10 == 0 or completed == len(tasks):
                        logger.info(
                            f"Parallel preprocessing completed {completed}/{len(tasks)} molecules."
                        )
        self.finalize_parallel_results(results)
    def finalize_parallel_results(self, results):
        self.smiles_list = []
        self.names_list = []
        self.labels = []
        for result in sorted(results, key=lambda item: item["idx"]):
            if result["data"] is None:
                continue
            self.data_list.append(_deserialize_worker_data(result["data"]))
            self.successful_labels.append(result["label"])
            self.successful_names.append(result["name"])
            self.successful_smiles.append(result["smiles"])
            self.smiles_list.append(result["smiles"])
            self.names_list.append(result["name"])
            self.labels.append(result["label"])
            self.global_dim = result["global_dim"]
            self.num_node_features = result["num_node_features"]
            self.edge_dim = result["edge_dim"]
            self.add_timing_totals(result.get("timing_totals", {}))
            self.processed_count += 1
    def preprocessing_cache_settings(self):
        return {
            "cache_version": 4,
            "node_block": self.node_block,
            "num_confs": self.num_confs,
            "max_isomers": self.max_isomers,
            "stereo_try_embedding": self.stereo_try_embedding,
            "embed_num_threads": self.embed_num_threads,
            "buried_volume_radius": self.buried_volume_radius,
            "buried_volume_grid_spacing": self.buried_volume_grid_spacing,
        }
    def molecule_cache_path(self, smiles):
        if not self.use_cache:
            return None
        smiles = _clean_smiles(smiles)
        if smiles is None:
            return None
        mol = Chem.MolFromSmiles(smiles)
        canonical_smiles = Chem.MolToSmiles(mol) if mol is not None else smiles
        cache_key_data = {
            "smiles": canonical_smiles,
            "settings": self.preprocessing_cache_settings(),
        }
        cache_key = hashlib.sha256(
            json.dumps(cache_key_data, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return os.path.join(self.cache_dir, f"{cache_key}.pt")
    def load_cached_molecule(self, smiles, name, label):
        cache_path = self.molecule_cache_path(smiles)
        if cache_path is None or not os.path.exists(cache_path):
            return None
        try:
            payload = torch.load(cache_path, weights_only=False)
            data = _deserialize_worker_data(payload["data"])
            data.smiles = smiles
            data.name = name
            if label is not None:
                data.y = torch.tensor([label], dtype=torch.float).reshape(-1, 1)
            elif hasattr(data, "y"):
                del data.y
            self.global_dim = data.u.size(-1)
            self.num_node_features = data.x.size(-1)
            self.edge_dim = data.edge_attr.size(-1)
            return data
        except Exception as e:
            logger.warning(f"Failed to load cached molecule for {smiles}: {e}")
            return None
    def save_cached_molecule(self, data, smiles):
        self.save_cached_payload(_serialize_data_for_worker(data, include_y=False), smiles)
    def save_cached_payload(self, data_payload, smiles):
        cache_path = self.molecule_cache_path(smiles)
        if cache_path is None:
            return
        os.makedirs(self.cache_dir, exist_ok=True)
        payload = {
            "data": {
                key: value for key, value in data_payload.items()
                if key not in {"y", "name"}
            },
            "global_dim": data_payload["u"].shape[-1],
            "num_node_features": data_payload["x"].shape[-1],
            "edge_dim": data_payload["edge_attr"].shape[-1],
            "settings": self.preprocessing_cache_settings(),
        }
        tmp_path = f"{cache_path}.tmp"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, cache_path)
    def prewarm_mendeleev_cache(self):
        stage_start = time.perf_counter()
        atomic_nums = _atomic_numbers_from_smiles(self.smiles_list)
        for atomic_num in atomic_nums:
            _get_element_props(atomic_num)
        self._mendeleev_cache = _MENDELEEV_CACHE
        logger.info(
            f"Prewarmed element feature cache for {len(atomic_nums)} atomic numbers "
            f"in {time.perf_counter() - stage_start:.3f}s."
        )
    def smiles_to_data(self, smiles, name, label=None, output_dir="molecule_images"):
        raw_smiles = smiles
        smiles = _clean_smiles(smiles)
        if smiles is None:
            logger.warning(
                f"Skipping molecule with missing/non-string SMILES: Name: {name}, SMILES: {raw_smiles}"
            )
            return None
        timings = {}
        total_start = time.perf_counter()
        debug_info = {
            "undefined_stereo": False,
            "enumerated_isomers": 0,
            "successful_conformers": 0,
        }
        try:
            stage_start = time.perf_counter()
            mol = Chem.MolFromSmiles(smiles)
            timings["parse_smiles"] = time.perf_counter() - stage_start
            if mol is None:
                logger.warning(f"Failed to parse SMILES: {smiles}")
                return None
            stage_start = time.perf_counter()
            mol = self.standardizer.standardize(mol)
            mol = self.correct_atom_types(mol)
            Chem.AssignStereochemistry(mol, force=False, cleanIt=True)
            timings["standardize_and_assign_stereo"] = time.perf_counter() - stage_start
            stage_start = time.perf_counter()
            chiral_centers = Chem.FindMolChiralCenters(
                mol, includeUnassigned=True, useLegacyImplementation=False
            )
            has_undefined_atom_stereo = any(tag == "?" for _, tag in chiral_centers)
            if has_undefined_atom_stereo:
                opts = StereoEnumerationOptions(
                    tryEmbedding=self.stereo_try_embedding,
                    unique=True,
                    onlyUnassigned=True,
                    maxIsomers=self.max_isomers,
                )
                candidate_mols = list(EnumerateStereoisomers(mol, options=opts))
                if not candidate_mols:
                    candidate_mols = [mol]
            else:
                candidate_mols = [mol]
            timings["enumerate_stereoisomers"] = time.perf_counter() - stage_start
            debug_info["undefined_stereo"] = has_undefined_atom_stereo
            debug_info["enumerated_isomers"] = len(candidate_mols)
            best_mol = None
            best_conf = None
            best_energy = np.inf
            timings["embed_conformers"] = 0.0
            timings["mmff_setup"] = 0.0
            timings["minimize_conformers"] = 0.0
            timings["select_best_conformer"] = 0.0
            for candidate in candidate_mols:
                try:
                    cand = Chem.Mol(candidate)
                    Chem.AssignStereochemistry(cand, force=False, cleanIt=True)
                    cand_h = Chem.AddHs(cand)
                    params = AllChem.ETKDGv3()
                    params.enforceChirality = True
                    params.useSmallRingTorsions = True
                    params.useBasicKnowledge = True
                    params.pruneRmsThresh = 0.5
                    params.numThreads = self.embed_num_threads
                    stage_start = time.perf_counter()
                    conf_ids = AllChem.EmbedMultipleConfs(
                        cand_h,
                        numConfs=self.num_confs,
                        params=params
                    )
                    timings["embed_conformers"] += time.perf_counter() - stage_start
                    if not conf_ids:
                        continue
                    stage_start = time.perf_counter()
                    mp = AllChem.MMFFGetMoleculeProperties(cand_h)
                    timings["mmff_setup"] += time.perf_counter() - stage_start
                    if mp is None:
                        continue
                    local_best_cid = None
                    local_best_energy = np.inf
                    successful_conf_count = 0
                    for cid in conf_ids:
                        try:
                            stage_start = time.perf_counter()
                            ff = AllChem.MMFFGetMoleculeForceField(cand_h, mp, confId=cid)
                            if ff is None:
                                timings["mmff_setup"] += time.perf_counter() - stage_start
                                continue
                            timings["mmff_setup"] += time.perf_counter() - stage_start
                            stage_start = time.perf_counter()
                            ff.Minimize()
                            e = ff.CalcEnergy()
                            timings["minimize_conformers"] += time.perf_counter() - stage_start
                            successful_conf_count += 1
                            if e < local_best_energy:
                                local_best_energy = e
                                local_best_cid = cid
                        except Exception:
                            continue
                    debug_info["successful_conformers"] += successful_conf_count
                    if local_best_cid is None:
                        continue
                    if local_best_energy < best_energy:
                        stage_start = time.perf_counter()
                        best_energy = local_best_energy
                        Chem.AssignStereochemistryFrom3D(
                            cand_h,
                            confId=local_best_cid,
                            replaceExistingTags=False
                        )
                        cand_no_h = Chem.RemoveHs(cand_h)
                        conf_h = cand_h.GetConformer(local_best_cid)
                        conf = Chem.Conformer(cand_no_h.GetNumAtoms())
                        for atom_id in range(cand_no_h.GetNumAtoms()):
                            pos = conf_h.GetAtomPosition(atom_id)
                            conf.SetAtomPosition(atom_id, pos)
                        best_mol = cand_no_h
                        best_conf = conf
                        timings["select_best_conformer"] += time.perf_counter() - stage_start
                except Exception as e:
                    logger.warning(f"Candidate stereoisomer failed for {smiles}: {e}")
                    continue
            if best_mol is None or best_conf is None:
                logger.error(f"No valid stereoisomer/conformer generated for: {smiles}")
                return None
            stage_start = time.perf_counter()
            best_mol.RemoveAllConformers()
            best_mol.AddConformer(best_conf)
            timings["attach_conformer"] = time.perf_counter() - stage_start
            stage_start = time.perf_counter()
            data = self.extract_features(best_mol, best_conf, name, smiles, label, output_dir)
            timings["extract_features_total"] = time.perf_counter() - stage_start
            timings["total"] = time.perf_counter() - total_start
            self.log_timing_summary(name, timings, debug_info, group="molecule")
            return data
        except Exception as e:
            logger.error(f"General failure processing SMILES: {smiles}, Error: {e}")
            return None
    def extract_features(self, mol, conf, name, smiles, label, output_dir):
        try:
            timings = {}
            stage_start = time.perf_counter()
            atom_features, atom_feature_timings = self.get_atom_features(
                mol,
                conf,
                molecule_name=name,
                return_timings=True,
            )
            timings["atom_features_total"] = time.perf_counter() - stage_start
            timings.update(atom_feature_timings)
            stage_start = time.perf_counter()
            edge_index, edge_attr = self.get_edge_index_and_features(mol, conf, self.node_block)
            timings["edge_features"] = time.perf_counter() - stage_start
            if edge_index is None or edge_attr is None:
                logger.error(f"Invalid edge features generated for molecule {name}")
                return None
            if edge_index.numel() > 0 and edge_index.max().item() >= atom_features.size(0):
                logger.error(f"Invalid edge index detected: {edge_index.max().item()} exceeds number of atoms {atom_features.size(0)}")
                return None
            stage_start = time.perf_counter()
            global_features = self.get_global_features(mol, conf)
            timings["global_features"] = time.perf_counter() - stage_start
            stage_start = time.perf_counter()
            if label is not None:
                target = torch.tensor([label], dtype=torch.float).reshape(-1, 1)
                data = Data(x=atom_features, edge_index=edge_index, edge_attr=edge_attr, u=global_features, y=target,)
            else:
                data = Data(x=atom_features, edge_index=edge_index, edge_attr=edge_attr, u=global_features)        
            data.smiles = smiles
            data.name = name
            timings["build_data_object"] = time.perf_counter() - stage_start
            stage_start = time.perf_counter()
            if self.save_images:
                self.save_molecule_image(mol, name, output_dir)
            timings["save_molecule_image"] = time.perf_counter() - stage_start
            self.log_timing_summary(f"{name} feature extraction", timings, group="feature_extraction")
            return data
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}, SMILES: {smiles}")
            return None
    def log_timing_summary(self, molecule_name, timings, debug_info=None, group="molecule"):
        if not timings:
            return
        bottleneck_timings = {
            stage: seconds for stage, seconds in timings.items() if stage != "total"
        }
        bottleneck_name, bottleneck_seconds = max(
            bottleneck_timings.items(), key=lambda item: item[1]
        )
        timing_text = ", ".join(
            f"{stage}={seconds:.6f}s" for stage, seconds in timings.items()
        )
        debug_text = ""
        if debug_info:
            debug_text = ", " + ", ".join(
                f"{key}={value}" for key, value in debug_info.items()
            )
        if not getattr(self, "suppress_timing_logs", False):
            logger.info(
                f"Timing summary for molecule {molecule_name}: {timing_text}. "
                f"Bottleneck: {bottleneck_name}={bottleneck_seconds:.6f}s{debug_text}."
            )
        self.add_timing_totals({group: timings})
    def add_timing_totals(self, timing_totals):
        for group, timings in timing_totals.items():
            group_totals = self.timing_totals.setdefault(group, {})
            for stage, seconds in timings.items():
                group_totals[stage] = group_totals.get(stage, 0.0) + seconds
    def log_aggregate_timing_summary(self):
        for group, timings in self.timing_totals.items():
            if not timings:
                continue
            bottleneck_timings = {
                stage: seconds for stage, seconds in timings.items() if stage != "total"
            }
            if not bottleneck_timings:
                bottleneck_timings = timings
            bottleneck_name, bottleneck_seconds = max(
                bottleneck_timings.items(), key=lambda item: item[1]
            )
            timing_text = ", ".join(
                f"{stage}={seconds:.6f}s" for stage, seconds in timings.items()
            )
            logger.info(
                f"Aggregate timing summary for {group}: {timing_text}. "
                f"Bottleneck: {bottleneck_name}={bottleneck_seconds:.6f}s."
            )
            if group == "feature_extraction" and bottleneck_name == "buried_volume":
                logger.info(
                    "Optimization hint: buried_volume is CPU-bound and independent "
                    "per molecule. Prefer multiprocessing across molecules with "
                    "ProcessPoolExecutor or multiprocessing.Pool, passing SMILES/name/"
                    "label into each worker and constructing RDKit mol/conformer objects "
                    "inside the worker."
                )
    def min_max_normalize(self, value, min_value, max_value):
        if max_value == min_value:
            logger.warning(f"Normalization skipped: min_value == max_value for {value}")
            return 0
        normalized = (value - min_value) / (max_value - min_value)
        return normalized
    def get_cached_element_props(self, atomic_num):
        if atomic_num not in self._mendeleev_cache:
            self._mendeleev_cache[atomic_num] = _get_element_props(atomic_num)
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
    def _get_sphere_offsets(self, radius=3.5, grid_spacing=0.5):
        """
        Cache the spherical grid once instead of rebuilding it for every atom.
        """
        key = (radius, grid_spacing)
        if not hasattr(self, "_sphere_offset_cache"):
            self._sphere_offset_cache = {}
        if key not in self._sphere_offset_cache:
            grid = np.arange(-radius, radius + grid_spacing, grid_spacing, dtype=np.float32)
            grid_points = np.array(np.meshgrid(grid, grid, grid), dtype=np.float32).reshape(3, -1).T
            sphere_mask = np.linalg.norm(grid_points, axis=1) <= radius
            self._sphere_offset_cache[key] = grid_points[sphere_mask]
        return self._sphere_offset_cache[key]
    def calculate_buried_volumes_all_atoms(
        self,
        mol,
        conf,
        radius=3.5,
        grid_spacing=0.5,
        unique_occupancy=True,
    ):
        """
        Vectorized buried-volume calculation for all atoms in one molecule.
        unique_occupancy=False preserves your current behavior:
            overlapping vdW volumes are double-counted.
        unique_occupancy=True gives a more physical occupied-grid fraction:
            each grid point is counted only once, even if multiple atoms occupy it.
        """
        pt = Chem.GetPeriodicTable()
        offsets = self._get_sphere_offsets(radius=radius, grid_spacing=grid_spacing)
        n_grid = len(offsets)
        coords = np.array(
            [
                [
                    conf.GetAtomPosition(i).x,
                    conf.GetAtomPosition(i).y,
                    conf.GetAtomPosition(i).z,
                ]
                for i in range(mol.GetNumAtoms())
            ],
            dtype=np.float32,
        )
        vdw_radii = np.array(
            [pt.GetRvdw(atom.GetAtomicNum()) for atom in mol.GetAtoms()],
            dtype=np.float32,
        )
        buried = np.zeros(mol.GetNumAtoms(), dtype=np.float32)
        for center_idx in range(mol.GetNumAtoms()):
            center = coords[center_idx]
            center_to_atom = np.linalg.norm(coords - center, axis=1)
            neighbor_mask = center_to_atom <= (radius + vdw_radii)
            neighbor_mask[center_idx] = False
            neighbor_coords = coords[neighbor_mask]
            neighbor_radii = vdw_radii[neighbor_mask]
            if len(neighbor_coords) == 0:
                buried[center_idx] = 0.0
                continue
            sphere_points = center + offsets
            diff = sphere_points[:, None, :] - neighbor_coords[None, :, :]
            dist2 = np.sum(diff * diff, axis=2)
            occupied = dist2 <= (neighbor_radii[None, :] ** 2)
            if unique_occupancy:
                buried[center_idx] = occupied.any(axis=1).sum() / n_grid
            else:
                buried[center_idx] = occupied.sum() / n_grid
        return buried
    def save_molecule_image(self, mol, name, output_dir="molecule_images", img_size=(200, 200)):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        img = Draw.MolToImage(mol, size=img_size)
        img_path = os.path.join(output_dir, f"molecule_{name}.png")
        img.save(img_path)          
    def hybridization_to_index(self, hybridization):
        return self.hybridization_dict.get(hybridization, 0)
    def get_atom_features(self, mol, conf, molecule_name=None, return_timings=False):
        timings = {}
        atom_features = []
        buried_volume_start = time.perf_counter()
        buried_volumes = self.calculate_buried_volumes_all_atoms(
            mol,
            conf,
            radius=self.buried_volume_radius,
            grid_spacing=self.buried_volume_grid_spacing,
            unique_occupancy=False,
        )
        timings["buried_volume"] = time.perf_counter() - buried_volume_start
        feature_assembly_start = time.perf_counter()
        heavy_atom_count = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 1:
                continue
            heavy_atom_count += 1
            idx = atom.GetIdx()
            atomic_num = atom.GetAtomicNum()
            props = self.get_cached_element_props(atomic_num)
            atom_feature = [
                (atomic_num - 1) / 178,
                float(buried_volumes[idx]),
                self.hybridization_dict.get(atom.GetHybridization(), 0),
                props["electronegativity"],
                props["polarizability"],
                props["vdw_radius"],
            ]
            atom_features.append(atom_feature)
        timings["atom_feature_assembly"] = time.perf_counter() - feature_assembly_start
        molecule_label = molecule_name if molecule_name is not None else "<unknown>"
        avg_time = timings["buried_volume"] / heavy_atom_count if heavy_atom_count else 0.0
        if not getattr(self, "suppress_timing_logs", False):
            logger.info(
                f"Buried volume timing for molecule {molecule_label}: "
                f"{timings['buried_volume']:.6f}s across {heavy_atom_count} heavy atoms "
                f"({avg_time:.6f}s/atom)."
            )
        self.num_node_features = len(atom_features[0]) if atom_features else 0
        atom_features_tensor = torch.tensor(atom_features, dtype=torch.float)
        if return_timings:
            return atom_features_tensor, timings
        return atom_features_tensor
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
        atom_index_map = {}
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 1:
                continue
            atom_index_map[atom.GetIdx()] = len(atom_index_map)
        try:
            for bond in mol.GetBonds():
                rdkit_i = bond.GetBeginAtomIdx()
                rdkit_j = bond.GetEndAtomIdx()
                if rdkit_i not in atom_index_map or rdkit_j not in atom_index_map:
                    continue
                i = atom_index_map[rdkit_i]
                j = atom_index_map[rdkit_j]
                bond_length = Chem.rdMolTransforms.GetBondLength(conf, rdkit_i, rdkit_j)
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
        self.edge_dim = len(edge_attr[0]) if edge_attr else 4
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, self.edge_dim), dtype=torch.float)
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
