from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from common.dataset import TMH
from common.models import CNN
from common.module import CaitModule


def train():
    logger = TensorBoardLogger(save_dir="logs")
    callbacks = [
        ModelCheckpoint(
            monitor="train_loss"
        ), # TODO: more params
        EarlyStopping(
            monitor="train_loss"
        ) # TODO: more params
    ]

    dataset = TMH(data_root=Path("/Users/fga/data/tmh")) # TODO: move to cfg
    model = CNN()
    module = CaitModule(model=model)

    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        gpus=torch.cuda.device_count(),
        min_epochs=1, # TODO: move to cfg
        max_epochs=5 # TODO: move to cfg
    )

    trainer.fit(module, dataset, ckpt_path=None) # TODO: move to cfg

def main():
    train()

if __name__ == '__main__':
    main()
