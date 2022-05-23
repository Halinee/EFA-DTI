from typing import Union

import dgl
import numpy as np
import torch as th
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


def dgl_graph(graph, n_data_key="feat", e_data_key="feat") -> dgl.DGLHeteroGraph:
    g = dgl.graph(tuple(graph["edge_index"]), num_nodes=graph["num_nodes"])
    if graph["edge_feat"] is not None:
        g.edata[e_data_key] = th.from_numpy(graph["edge_feat"])
    if graph["node_feat"] is not None:
        g.ndata[n_data_key] = th.from_numpy(graph["node_feat"])
    return g


def get_fingerprint(
    mol: Union[Chem.Mol, str], r=3, n_bits=2048, **kwargs
) -> np.ndarray:
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, r, nBits=n_bits, **kwargs)
    arr = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr
