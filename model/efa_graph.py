import math
from typing import Dict

import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
from dgl.ops import edge_softmax
from einops import rearrange
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from norm import ReZero, GraphNormAndProj, EdgeNormWithGainAndBias
from pooling import DeepSet, MeanMaxPool


class ActGLU(nn.Module):
    def __init__(self, d_in: int, d_out: int, act: str = "relu"):
        super(ActGLU, self).__init__()
        self.proj = nn.Linear(d_in, d_out * 2)
        acts = {"gelu": nn.GELU, "relu": nn.ReLU}
        self.act = acts[act]()

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.act(gate)


class GraphNetBlock(nn.Module):
    def __init__(self, graph_params: Dict):
        """
        Initialize a multi-headed attention block compatible with DGLGraph
        inputs. Given a fully connected input model with self loops,
        is analogous to original Transformer.

        Args:
            d_in: input dimension
            qk_dim: head dimension
            n_heads: number of heads
            dropout: dropout probability
            attn_weight_norm: attention pooling method, 'norm' or 'softmax'
        """
        super(GraphNetBlock, self).__init__()
        self.in_dim = graph_params["mol_dim"]
        self.qk_dim = int(self.in_dim // self.in_dim)
        self.v_dim = max(64, int(self.in_dim // self.in_dim))
        self.n_heads = graph_params["mol_n_heads"]
        self.d_hidden = self.n_heads * self.v_dim

        self.attn_weight_norm = {
            "norm": EdgeNormWithGainAndBias(self.n_heads),
            "softmax": edge_softmax,
        }[graph_params["attn_weight_norm"]]
        self.attn_dropout = nn.Dropout(graph_params["dropout"])

        def pwff():
            return nn.Sequential(
                ActGLU(self.in_dim, self.in_dim * 2, graph_params["act"]),
                nn.Dropout(self.in_dim["dropout"]),
                nn.Linear(self.in_dim * 2, self.in_dim),
            )

        self.node_rezero = ReZero()
        self.edge_rezero = ReZero()
        self.node_ff = pwff()
        self.edge_ff = pwff()
        self.node_ff2 = pwff()
        self.edge_ff2 = pwff()

        self.q_proj = nn.Linear(self.in_dim, self.qk_dim * self.n_heads)
        self.k_proj = nn.Linear(self.in_dim, self.qk_dim * self.n_heads)
        self.v_proj = nn.Linear(self.in_dim, self.v_dim * self.n_heads)
        self.eq_proj = nn.Linear(self.in_dim, self.qk_dim * self.n_heads)
        self.ek_proj = nn.Linear(self.in_dim, self.qk_dim * self.n_heads)
        self.ev_proj = nn.Linear(self.in_dim, self.v_dim * self.n_heads)

        self.mix_nodes = GraphNormAndProj(
            d_in=self.n_heads * self.v_dim,
            d_out=self.in_dim,
            act=graph_params["act"],
            dropout=graph_params["dropout"],
            norm_type=graph_params["norm_type"],
        )

    def forward(self, g: dgl.DGLGraph, n: th.Tensor, e: th.Tensor):
        # convection
        n = n + self.node_rezero(self.node_ff(n))
        e = e + self.edge_rezero(self.edge_ff(e))

        # diffusion (attn)
        q = rearrange(self.q_proj(n), "b (h qk) -> b h qk", h=self.n_heads)
        k = rearrange(self.k_proj(n), "b (h qk) -> b h qk", h=self.n_heads)
        v = rearrange(self.v_proj(n), "b (h v) -> b h v", h=self.n_heads)
        eq = rearrange(self.eq_proj(e), "b (h qk) -> b h qk", h=self.n_heads)
        ek = rearrange(self.ek_proj(e), "b (h qk) -> b h qk", h=self.n_heads)
        ev = rearrange(self.ev_proj(e), "b (h v) -> b h v", h=self.n_heads)

        g.ndata.update({"q": q, "k": k, "v": v})
        g.edata.update({"eq": eq, "ek": ek, "ev": ev})

        g.apply_edges(fn.v_dot_u("q", "k", "n2n"))  # n2n
        g.apply_edges(fn.v_dot_e("q", "ek", "n2e"))  # n2e
        g.apply_edges(fn.e_dot_u("eq", "k", "e2n"))  # e2n
        if self.attn_weight_norm == "softmax":
            scale = math.sqrt(self.qk_dim)
            g.edata["n2n"] /= scale
            g.edata["n2e"] /= scale
            g.edata["e2n"] /= scale
        n2n_attn = self.attn_dropout(self.attn_weight_norm(g, g.edata["n2n"]))
        n2e_attn = self.attn_dropout(self.attn_weight_norm(g, g.edata["n2e"]))
        e2n_attn = self.attn_dropout(self.attn_weight_norm(g, g.edata["n2e"]))

        # aggregate normalized weighted values per node
        g.apply_edges(
            lambda edge: {
                "wv": n2n_attn * edge.src["v"]
                + n2e_attn * edge.data["ev"]
                + e2n_attn * edge.src["v"]
            }
        )
        g.update_all(fn.copy_e("wv", "wv"), fn.sum("wv", "z"))

        n = n + self.node_rezero(
            self.mix_nodes(g.ndata["z"].view(-1, self.d_hidden), g.batch_num_nodes())
        )

        # convection
        n = n + self.node_rezero(self.node_ff2(n))
        e = e + self.edge_rezero(self.edge_ff2(e))

        return g, n, e


class GraphNet(nn.Module):
    def __init__(self, graph_params: Dict):
        super(GraphNet, self).__init__()
        self.dim = graph_params["mol_dim"]
        self.n_layers = graph_params["mol_n_layers"]
        self.atom_enc = AtomEncoder(self.dim)
        self.bond_enc = BondEncoder(self.dim)
        self.attn_layers = nn.ModuleList()

        for _ in range(self.n_layers):
            self.attn_layers.append(GraphNetBlock(graph_params))

        if graph_params["pool_type"] == "deepset":
            self.readout = DeepSet(self.dim, self.dim, dropout=graph_params["dropout"])
        elif graph_params["pool_type"] == "mean_max":
            self.readout = MeanMaxPool(self.dim * 2)

    def forward(self, g: dgl.DGLGraph):
        n = self.atom_enc(g.ndata["feat"])
        e = self.bond_enc(g.edata["feat"])
        for i in range(self.n_layers):
            g, n, e = self.attn_layers[i](g, n, e)
        out = self.readout(g, n)
        return out
