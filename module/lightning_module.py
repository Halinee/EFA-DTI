from typing import Dict

import pytorch_lightning as pl
import torch as th
import torch.nn.functional as F
from adabelief_pytorch import AdaBelief
from lifelines.utils import concordance_index
from sklearn.metrics import r2_score

from model.mlp import MLP
from model.efa_graph import GraphNet


class DTIModule(pl.LightningModule):
    def __init__(self, module_params: Dict):
        super(DTIModule, self).__init__()
        self.save_hyperparameters()

        self.mol_enc = GraphNet(module_params["graph_params"])
        self.fingerprint_enc = MLP(module_params["fingerprint_params"])
        self.prottrans_enc = MLP(module_params["prottrans_params"])
        out_dim = (
            self.mol_enc.readout.out_features
            + self.fingerprint_enc[-1].out_features
            + self.prottrans_enc[-1].out_features
        )
        module_params["output_params"]["output_dims"] = (
            out_dim + module_params["output_params"]["output_dims"]
        )
        self.output = MLP(module_params["output_params"])

    def forward(self, g, fp, pt):
        g_emb = self.mol_enc(g)
        fp_emb = self.fingerprint_enc(fp)
        pt_emb = self.prottrans_enc(pt)

        return self.output(th.cat([g_emb, fp_emb, pt_emb], -1))

    def sharing_step(self, batch, _=None):
        y = batch[-1]
        g, fp, pt, _ = batch
        yhat = self(g, fp, pt).squeeze()

        return yhat, y

    def training_step(self, batch, _=None):
        yhat, y = self.sharing_step(batch)
        mse = F.mse_loss(yhat, y)
        self.log("train_mse", mse)

        return mse

    def validation_step(self, batch, _=None):
        y, yhat = self.sharing_step(batch)

        return {"yhat": yhat, "y": y}

    def validation_epoch_end(self, outputs):
        yhats = []
        ys = []
        for o in outputs:
            yhats.append(o["yhat"])
            ys.append(o["y"])
        yhat = th.cat(yhats).detach().cpu()
        y = th.cat(ys).detach().cpu()

        self.log_dict(
            {
                "valid_mse": th.as_tensor(F.mse_loss(yhat, y), device=self.device),
                "valid_ci": th.as_tensor(
                    concordance_index(y, yhat), device=self.device
                ),
                "valid_r2": th.as_tensor(r2_score(y, yhat), device=self.device),
            }
        )

    def configure_optimizers(self):
        optimizer = AdaBelief(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=float(self.hparams.weight_decay),
            eps=float(self.hparams.eps),
        )
        scheduler = {
            "scheduler": th.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda epoch: max(
                    1e-7, 1 - epoch / self.hparams.lr_anneal_epochs
                ),
            ),
            "reduce_on_plateau": False,
            "interval": "epoch",
            "frequency": 1,
        }

        return [optimizer], [scheduler]
