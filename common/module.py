from pathlib import Path

import torch
import torch.nn.functional as F
from pycm import ConfusionMatrix
from pytorch_lightning import LightningModule
from torch.optim import Adam

from common.cait_models import CaiT


class CaitModule(LightningModule):
    def __init__(self, cfg, model):
        super().__init__()
        self.cfg = cfg
        self.model = model

    def _generic_step(self, batch, mode):
        embed, label = batch

        if isinstance(self.model, CaiT):
            y_hat, attn_weights = self.model(embed)
        else:
            y_hat = self.model(embed)

        if self.cfg.weighted_loss:
            loss = F.cross_entropy(y_hat, label, weight=self.trainer.datamodule.weights.to(y_hat.device))
        else:
            loss = F.cross_entropy(y_hat, label)

        self.log(f"{mode}_loss", loss)
        return {
            f"{mode}_loss": loss,
            f"{mode}_pred": y_hat.detach().cpu(),
            f"{mode}_label": label.detach().cpu()
        }

    def _generic_epoch_end(self, outputs, mode):
        preds = torch.cat([d[f"{mode}_pred"].argmax(dim=1) for d in outputs]).int().numpy()
        labels = torch.cat([d[f"{mode}_label"] for d in outputs]).int().numpy()
        cm = ConfusionMatrix(actual_vector=labels, predict_vector=preds)
        metrics = {
            f"{mode}_acc": cm.ACC_Macro,
            f"{mode}_tpr_micro": cm.TPR_Micro,
            f"{mode}_tpr_macro": cm.TPR_Macro,
            f"{mode}_ppv_micro": cm.PPV_Micro,
            f"{mode}_ppv_macro": cm.PPV_Macro
        }

        for key, value in metrics.items():
            if value != 'None':
                self.log(key, value)

        return metrics

    def training_step(self, batch, batch_idx):
        metrics = self._generic_step(batch, "train")
        return metrics["train_loss"]

    def training_epoch_end(self, outputs):
        # nothing to do here since validation starts automatically
        pass

    def validation_step(self, batch, batch_idx):
        return self._generic_step(batch, "val")

    def validation_epoch_end(self, outputs):
        self._generic_epoch_end(outputs, "val")

    def test_step(self, batch, batch_idx):
        return self._generic_step(batch, "test")

    def test_epoch_end(self, outputs):
        torch.save(outputs, Path(self.cfg.logdir) / f"{self.cfg.model}-test_outputs.pkl")
        print(f"saved test outputs to {self.cfg.logdir}")
        return self._generic_epoch_end(outputs, "test")

    def on_save_checkpoint(self, checkpoint):
        print(f"Saving checkpoint at the end of epoch {self.current_epoch}")

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3) # TODO: move to cfg