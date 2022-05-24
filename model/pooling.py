import dgl
import torch as th
import torch.nn as nn


class DeepSet(nn.Module):
    def __init__(self, d_in: int, d_out: int, dropout: float = 0.1):
        super(DeepSet, self).__init__()
        self.out_features = d_out
        self.glu = nn.Sequential(nn.Linear(d_in, d_in * 2), nn.GLU())
        self.agg = nn.Sequential(
            nn.BatchNorm1d(d_in), nn.Dropout(dropout), nn.Linear(d_in, d_out)
        )

    def forward(self, g, n):
        g.ndata["out"] = self.glu(n)
        readout = self.agg(dgl.readout_nodes(g, "out", op="sum"))

        return readout


class MeanMaxPool(nn.Module):
    def __init__(self, dim: int):
        super(MeanMaxPool, self).__init__()
        self.out_features = dim
        self.gain = nn.Parameter(th.ones(dim))
        self.bias = nn.Parameter(th.zeros(dim))

    def forward(self, g, n, key="out"):
        g.ndata[key] = n
        max = dgl.readout_nodes(g, key, op="max")
        mean = dgl.readout_nodes(g, key, op="mean")
        out = th.cat([max, mean], dim=-1)
        return out * self.gain + self.bias
