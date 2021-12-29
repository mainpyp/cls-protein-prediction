import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.optim import Adam


class CaitModule(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def _generic_step(self, batch, mode):
        embed, label = batch
        expand_channel = embed.unsqueeze(1)
        y_hat = self.model(expand_channel)
        loss = F.cross_entropy(y_hat, label.float()) # TODO: weight?
        # TODO: acc?
        self.log(f"{mode}_loss", loss)

        return {
            "loss": loss
        }

    def training_step(self, batch, batch_idx):
        metrics = self._generic_step(batch, "train")
        return metrics["loss"]

    def validation_step(self, batch, batch_idx):
        metrics = self._generic_step(batch, "val")
        return metrics["loss"]

    def validation_epoch_end(self, outputs):
        # TODO:
        pass

    def test_step(self, batch, batch_idx):
        metrics = self._generic_step(batch, "test")
        return metrics["loss"]

    def test_epoch_end(self, outputs):
        # TODO:
        pass

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3) # TODO: move to cfg