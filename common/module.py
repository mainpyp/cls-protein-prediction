import torch
import torch.nn.functional as F
from pycm import ConfusionMatrix
from pytorch_lightning import LightningModule
from torch.optim import Adam


class CaitModule(LightningModule):
    def __init__(self, cfg, model):
        super().__init__()
        self.cfg = cfg
        self.model = model

    def _generic_step(self, batch, mode):
        embed, label = batch
        expand_channel = embed.unsqueeze(1)
        y_hat = self.model(expand_channel)
        loss = F.cross_entropy(y_hat, label) # TODO: weight?
        self.log(f"{mode}_loss", loss)
        return {
            f"{mode}_loss": loss,
            f"{mode}_pred": y_hat,
            f"{mode}_label": label
        }

    def _generic_epoch_end(self, outputs, mode):
        preds = torch.sigmoid(torch.cat([d[f"{mode}_pred"].argmax(dim=1) for d in outputs]).detach().cpu()).int().numpy()
        labels = torch.cat([d[f"{mode}_label"] for d in outputs]).detach().cpu().int().numpy()
        cm = ConfusionMatrix(actual_vector=labels, predict_vector=preds)
        # TODO: log cm image
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
        return self._generic_epoch_end(outputs, "test")

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3) # TODO: move to cfg