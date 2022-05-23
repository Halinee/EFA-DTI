import os
import pickle
from typing import Callable

import pandas as pd
import torch as th
from ogb.utils import smiles2graph
from torch.utils.data import Dataset
from tqdm import tqdm

from transform import ligand, target


class DTIDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        data_name: str,
        reset: bool = False,
        device: int = 0,
        y_transform: Callable = None,
    ):
        self.reset = reset
        self.device = device
        self.y_transform = y_transform

        data_path = os.path.join(data_dir, data_name)
        if data_name[-3:] == "ftr":
            self.data = pd.read_feather(data_path)
        elif data_name[-3:] == "csv":
            self.data = pd.read_csv(data_path)
        else:
            raise ValueError(f"Invalid File Format : {data_name[-4:]}")

        # Graphs(g_emb)
        self.ligand_graphs = self.load_graph(data_dir)
        # Fingerprint(fp_emb)
        self.ligand_fps = self.load_fingerprint(data_dir)
        # ProtTrans(pt_emb)
        self.target_pts = self.load_prottrans(data_dir)

    def __getitem__(self, idx):
        # Raw dataset
        smiles = self.data["SMILES"][idx]
        sequence = self.data["SEQUENCE"][idx]
        y = self.data["IC50"][idx]

        # Preprocessed dataset
        g = self.ligand_graphs[smiles]
        fp = th.as_tensor(self.ligand_fps[smiles], dtype=th.float32).unsqueeze(0)
        pt = th.as_tensor(self.target_pts[sequence], dtype=th.float32).unsqueeze(0)
        y = th.as_tensor(y, dtype=th.float32).unsqueeze(0)

        if self.y_transform is not None:
            y = self.y_transform(y)

        return g, fp, pt, y

    def __len__(self):
        return len(self.data)

    def load_graph(self, path):
        graphs_path = os.path.join(path, "ligand_graphs.pkl")
        if not os.path.exists(graphs_path) or self.reset:
            print(
                f"{graphs_path} does not exist!\nProcessing SMILES to graphs...",
                flush=True,
            )
            graphs = {}
            for s in tqdm(self.data["SMILES"]):
                if not s in graphs:
                    graphs[s] = ligand.dgl_graph(smiles2graph(s))
            with open(graphs_path, "wb") as f:
                pickle.dump(graphs, f)
        else:
            print("Loading preprocessed graphs...", flush=True)
            with open(graphs_path, "rb") as f:
                graphs = pickle.load(f)

        return graphs

    def load_fingerprint(self, path):
        fingerprint_path = os.path.join(path, "ligand_fingerprints.pkl")
        if not os.path.exists(fingerprint_path) or self.reset:
            print(
                f"{fingerprint_path} does not exist!\nProcessing SMILES to fingerprints...",
                flush=True,
            )
            fingerprints = {}
            for s in tqdm(self.data["SMILES"]):
                if not s in fingerprints:
                    fingerprints[s] = ligand.get_fingerprint(s)
            with open(fingerprint_path, "wb") as f:
                pickle.dump(fingerprints, f)
        else:
            print("Loading preprocessed fingerprints...", flush=True)
            with open(fingerprint_path, "rb") as f:
                fingerprints = pickle.load(f)

        return fingerprints

    def load_prottrans(self, path):
        prottrans_path = os.path.join(path, "target_prottrans.pkl")
        if not os.path.exists(prottrans_path) or self.reset:
            print(
                f"{prottrans_path} does not exist!\nProcessing proteins to ProtTrans embedding...",
                flush=True,
            )
            pt_emb = target.EmbedProt()
            prottrans = {}
            for s in tqdm(self.data["SEQUENCE"]):
                if not s in prottrans:
                    prottrans[s] = pt_emb([s], device=self.device)[0]
            with open(prottrans_path, "wb") as f:
                pickle.dump(prottrans, f)
        else:
            print("Loading preprocessed ProtTrans embedding...", flush=True)
            with open(prottrans_path, "rb") as f:
                prottrans = pickle.load(f)

        return prottrans
