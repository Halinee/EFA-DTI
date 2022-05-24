from typing import List

import torch.nn as nn


class MLP(nn.Sequential):
    def __init__(
        self, dims: List, norm: bool = True, dropout: float = 0.1, act: str = "relu"
    ):
        acts = {"gelu": nn.GELU, "relu": nn.ReLU}
        act = acts[act]
        l = []
        for i in range(len(dims) - 2):
            l.extend(
                [
                    nn.Linear(dims[i], dims[i + 1]),
                    act(),
                    nn.BatchNorm1d(dims[i + 1]) if norm else nn.Identity(),
                    nn.Dropout(dropout),
                ]
            )
        l.append(nn.Linear(dims[-2], dims[-1]))
        super(MLP, self).__init__(*l)
