import hydra
import omegaconf
import torch
import pandas as pd
from omegaconf import ValueNode
from torch.utils.data import Dataset
import os
from torch_geometric.data import Data
import pickle
import numpy as np

from diffcsp.common.utils import PROJECT_ROOT
from diffcsp.common.data_utils import (
    preprocess, preprocess_tensors, preprocess_pdbs, add_scaled_lattice_prop)

def safe_tensor(x, dtype=None):
    # If 'x' is already a torch.Tensor, just clone/detach and optionally cast dtype
    if isinstance(x, torch.Tensor):
        out = x.clone().detach()
        if dtype is not None:
            out = out.to(dtype)
        return out
    else:
        # Otherwise, assume it's NumPy or list
        return torch.tensor(x, dtype=dtype)

class PairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_aa':
            return self.frac_coords_aa.size(0)
        if key == 'edge_index_cg':
            return self.frac_coords_cg.size(0)
        if key == 'bead_mapping':
            return self.num_atoms_aa
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        # Telling PyG "Do NOT concatenate this attribute"
        if key == 'bead_mapping':
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)

class DatasetPDBFiles(Dataset):
    def __init__(self, name: ValueNode, folder_path: ValueNode, prop: ValueNode,
                 preprocess_workers: ValueNode, lattice_scale_method: ValueNode,
                 save_path: ValueNode, scale: ValueNode, **kwargs):
        super().__init__()
        self.folder_path = folder_path
        self.prop = prop
        self.scale = scale
        self.preprocess_pdbs(save_path, preprocess_workers, scale, **kwargs)
        add_scaled_lattice_prop(self.cached_data, lattice_scale_method, scale=scale)

    def preprocess_pdbs(self, save_path, preprocess_workers, scale, **kwargs):
        if os.path.exists(save_path):
            self.cached_data = torch.load(save_path)
        else:
            cached_data = preprocess_pdbs(
                self.folder_path, preprocess_workers,
                smiles=kwargs['smiles'], num_mols=kwargs['num_mols'], same=kwargs['same'], scale=scale,
            )
            torch.save(cached_data, save_path)
            self.cached_data = cached_data

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        if self.scale == 'dual':
            graph_arrays_aa = data_dict['graph_arrays_aa']  # (frac_coords_aa, ...)
            graph_arrays_cg = data_dict['graph_arrays_cg']  # (frac_coords_cg, ...)
            bead_mapping = data_dict['bead_mapping']

            (frac_coords_aa,
             atom_types_aa,
             lengths,
             angles,
             atom_features_aa,
             edge_index_aa,
             edge_attr_aa,
             num_atoms_aa) = graph_arrays_aa

            (frac_coords_cg,
             atom_types_cg,
             _lengths2,
             _angles2,
             atom_features_cg,
             edge_index_cg,
             edge_attr_cg,
             num_atoms_cg) = graph_arrays_cg

            # Convert bead_mapping to a list of 1D LongTensors:
            # bead_mapping_list = [
            #     torch.tensor(sublist, dtype=torch.long) for sublist in bead_mapping
            # ]
            bead_mapping_list = [
                safe_tensor(sublist, dtype=torch.long) for sublist in bead_mapping
            ]

            # Cast everything to float32 (or long for indices) BEFORE creating PairData
            pair_data = PairData(
                frac_coords_aa=safe_tensor(frac_coords_aa, dtype=torch.float32),
                atom_types_aa=safe_tensor(atom_types_aa, dtype=torch.long),
                lengths=safe_tensor(lengths, dtype=torch.float32).view(1, -1),
                angles=safe_tensor(angles, dtype=torch.float32).view(1, -1),
                num_atoms_aa=num_atoms_aa,
                atom_features_aa=safe_tensor(atom_features_aa, dtype=torch.float32),
                edge_index_aa=safe_tensor(edge_index_aa, dtype=torch.long),
                edge_attr_aa=safe_tensor(edge_attr_aa, dtype=torch.float32),

                frac_coords_cg=safe_tensor(frac_coords_cg, dtype=torch.float32),
                atom_types_cg=safe_tensor(atom_types_cg, dtype=torch.long),
                num_atoms_cg=num_atoms_cg,
                atom_features_cg=safe_tensor(atom_features_cg, dtype=torch.float32),
                edge_index_cg=safe_tensor(edge_index_cg, dtype=torch.long),
                edge_attr_cg=safe_tensor(edge_attr_cg, dtype=torch.float32),

                bead_mapping=bead_mapping_list,
                # Optional: set num_nodes for PyG
                # num_nodes=num_atoms_aa + num_atoms_cg,
            )
            return pair_data

        else:
            # Single-scale
            frac_coords, atom_types, lengths, angles, atom_features, \
                edge_index, edge_attr, num_atoms = data_dict['graph_arrays']

            data = Data(
                frac_coords=safe_tensor(frac_coords, dtype=torch.float32),
                atom_types=safe_tensor(atom_types, dtype=torch.long),
                lengths=safe_tensor(lengths, dtype=torch.float32).view(1, -1),
                angles=safe_tensor(angles, dtype=torch.float32).view(1, -1),
                num_atoms=num_atoms,
                # For PyG to handle batching:
                num_nodes=num_atoms,
                atom_features=safe_tensor(atom_features, dtype=torch.float32),
                edge_index=safe_tensor(edge_index, dtype=torch.long),
                edge_attr=safe_tensor(edge_attr, dtype=torch.float32),
            )
            return data

    def __repr__(self) -> str:
        return f"DatasetPDBFiles({self.folder_path=})"

class CrystDataset(Dataset):
    def __init__(self, name: ValueNode, path: ValueNode,
                 prop: ValueNode, niggli: ValueNode, primitive: ValueNode,
                 graph_method: ValueNode, preprocess_workers: ValueNode,
                 lattice_scale_method: ValueNode, save_path: ValueNode, tolerance: ValueNode, use_space_group: ValueNode, use_pos_index: ValueNode,
                 **kwargs):
        super().__init__()
        self.path = path
        self.name = name
        self.df = pd.read_csv(path)
        self.prop = prop
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method
        self.use_space_group = use_space_group
        self.use_pos_index = use_pos_index
        self.tolerance = tolerance

        self.preprocess(save_path, preprocess_workers, prop)

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

    def preprocess(self, save_path, preprocess_workers, prop):
        if os.path.exists(save_path):
            self.cached_data = torch.load(save_path)
        else:
            cached_data = preprocess(
            self.path,
            preprocess_workers,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method,
            prop_list=[prop],
            use_space_group=self.use_space_group,
            tol=self.tolerance)
            torch.save(cached_data, save_path)
            self.cached_data = cached_data

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        # scaler is set in DataModule set stage
        prop = self.scaler.transform(data_dict[self.prop])
        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']

        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
            y=prop.view(1, -1),
        )

        if self.use_space_group:
            data.spacegroup = torch.LongTensor([data_dict['spacegroup']])
            data.ops = torch.Tensor(data_dict['wyckoff_ops'])
            data.anchor_index = torch.LongTensor(data_dict['anchors'])

        if self.use_pos_index:
            pos_dic = {}
            indexes = []
            for atom in atom_types:
                pos_dic[atom] = pos_dic.get(atom, 0) + 1
                indexes.append(pos_dic[atom] - 1)
            data.index = torch.LongTensor(indexes)
        return data

    def __repr__(self) -> str:
        return f"CrystDataset({self.name=}, {self.path=})"


class TensorCrystDataset(Dataset):
    def __init__(self, crystal_array_list, niggli, primitive,
                 graph_method, preprocess_workers,
                 lattice_scale_method, **kwargs):
        super().__init__()
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method

        self.cached_data = preprocess_tensors(
            crystal_array_list,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method)

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']

        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
        )
        return data

    def __repr__(self) -> str:
        return f"TensorCrystDataset(len: {len(self.cached_data)})"


@hydra.main(config_path=str("/home/gridsan/sakshay/experiments/flowmm/scripts_model/conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    from torch_geometric.data import Batch
    from diffcsp.common.data_utils import get_scaler_from_data_list
    dataset: DatasetPDBFiles = hydra.utils.instantiate(
        cfg.data.datamodule.datasets.train, _recursive_=False
    )
    # dataset: CrystDataset = hydra.utils.instantiate(
    #     cfg.data.datamodule.datasets.train, _recursive_=False
    # )
    # lattice_scaler = get_scaler_from_data_list(
    #     dataset.cached_data,
    #     key='scaled_lattice')
    # scaler = get_scaler_from_data_list(
    #     dataset.cached_data,
    #     key=dataset.prop)

    dataset.lattice_scaler = lattice_scaler
    dataset.scaler = scaler
    data_list = [dataset[i] for i in range(len(dataset))]
    batch = Batch.from_data_list(data_list)
    return batch


if __name__ == "__main__":
    main()
