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

            # from https://www.kaggle.com/piantic/vision-transformer-vit-visualize-attention-map/notebook
            render_attn_map = False
            if render_attn_map:
                att_mat = torch.stack(attn_weights).squeeze(1)
                att_mat = torch.mean(att_mat, dim=1)

                # To account for residual connections, we add an identity matrix to the
                # attention matrix and re-normalize the weights.
                residual_att = torch.eye(att_mat.size(1))
                aug_att_mat = att_mat + residual_att
                aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

                # Recursively multiply the weight matrices
                joint_attentions = torch.zeros(aug_att_mat.size())
                joint_attentions[0] = aug_att_mat[0]

                for n in range(1, aug_att_mat.size(0)):
                    joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

                v = joint_attentions[-1]
                # TODO: render attn map

        else:
            y_hat = self.model(embed)
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

    def on_save_checkpoint(self, checkpoint):
        print(f"Saving checkpoint at the end of epoch {self.current_epoch}")

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3) # TODO: move to cfg