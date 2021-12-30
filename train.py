import configargparse
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from common.cait_models import CaiT
from common.dataset import TMH
from common.models import CNN, MLP
from common.module import CaitModule


def load_cfg():
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add('-c', is_config_file=True, help='config file path')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    mode_group = parser.add_argument_group(title='Mode options')
    mode_group.add_argument("--mode", type=str, default='TRAIN')
    mode_group.add_argument("--on_cluster", action="store_true")
    mode_group.add_argument("--logdir", type=str, default="logs")
    mode_group.add_argument("--checkpoint", type=str, default="")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TRAINER ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    trainer_group = parser.add_argument_group(title='Trainer options')
    trainer_group.add_argument("--min_epochs", default=5, type=int)
    trainer_group.add_argument("--max_epochs", default=500, type=int)
    trainer_group.add_argument("--early_stopping_metric", type=str, default="val_acc")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TRAINING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    training_group = parser.add_argument_group(title='Training options')
    training_group.add_argument("--optimizer", type=str, default='Adam')
    training_group.add_argument("--learning_rate", type=float, default=1e-3)
    training_group.add_argument('--no-weighted_loss', dest='weighted_loss', action='store_false')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DATALOADER ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    dataloader_group = parser.add_argument_group(title='Dataloader options')
    dataloader_group.add_argument("--num_workers", type=int, default=0)
    dataloader_group.add_argument("--batch_size", type=int, default=2)
    dataloader_group.add_argument("--data_root", type=str, default=".")
    dataloader_group.add_argument("--num_classes", type=int, default=4)
    dataloader_group.add_argument("--dataset_val_percentage", type=float, default=0.1)
    dataloader_group.add_argument("--dataset_test_percentage", type=float, default=0.1)
    dataloader_group.add_argument("--reload_data", action="store_true")
    dataloader_group.add_argument("--balance_data", action="store_true")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MODEL ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    model_group = parser.add_argument_group(title='Model options')
    model_group.add_argument("--model", type=str, default="CNN")
    model_group.add_argument("--input_dim", type=int, default=25)
    model_group.add_argument("--hidden_dims_list", type=str, default="20,10")
    model_group.add_argument("--dropout_p", type=float, default=0.3)

    # pass 1: get all the parameters in the base config
    parser.set_defaults(on_cluster=False, weighted_loss=True, reload_data=False, balance_data=False)
    cfg, _ = parser.parse_known_args()

    return cfg

def train(cfg):
    logger = TensorBoardLogger(save_dir=cfg.logdir)
    callbacks = [
        ModelCheckpoint(
            monitor=cfg.early_stopping_metric
        ),
        EarlyStopping(
            monitor=cfg.early_stopping_metric
        )
    ]

    dataset = TMH(cfg=cfg)
    print(f"loading model {cfg.model}...")
    if cfg.model == "MLP":
        model = MLP(cfg=cfg)
    elif cfg.model == "CNN":
        model = CNN(cfg=cfg)
    elif cfg.model == "CaiT":
        model = CaiT()
    else:
        raise RuntimeError(f"Unsupported model {cfg.model}")
    print("loading module...")
    module = CaitModule(cfg=cfg, model=model)

    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        gpus=torch.cuda.device_count(),
        min_epochs=cfg.min_epochs,
        max_epochs=cfg.max_epochs
    )

    print("start training...")
    trainer.fit(module, dataset, ckpt_path=None if cfg.checkpoint == "" else cfg.checkpoint)

    print("start testing...")
    trainer.test(module, dataset)

def main():
    # TODO: do multiple runs to compare MLP vs CNN vs CaiT?
    cfg = load_cfg()
    train(cfg)

if __name__ == '__main__':
    main()
