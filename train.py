import configargparse
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

import wandb
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

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TRAINING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    training_group = parser.add_argument_group(title='Training options')
    training_group.add_argument("--gpus", type=int, default=-1)
    training_group.add_argument("--early_stopping_metric", type=str, default="val_acc")
    training_group.add_argument("--optimizer", type=str, default='Adam')
    training_group.add_argument("--learning_rate", type=float, default=1e-3)
    training_group.add_argument("--checkpoint", type=str, default="")
    training_group.add_argument('--no-weighted_loss', dest='weighted_loss', action='store_false')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DATALOADER ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    dataloader_group = parser.add_argument_group(title='Dataloader options')
    dataloader_group.add_argument("--num_workers", type=int, default=0)
    dataloader_group.add_argument("--batch_size", type=int, default=32)
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
    # TODO: add options for CNN, MLP, CaiT

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PREDICTION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    parser.add_argument("--fasta", type=str, help="path_to_fasta")
    parser.add_argument("--emb", type=str, help="path_to_embeddings")
    parser.add_argument("--output", type=str, help="path_to_output", default="./output.tsv")
    parser.add_argument(
        "--model_type",
        type=str,
        default="CAIT",
        choices=["CNN", "MLP", "CAIT"],
        help="Model type. Default CAIT",
    )

    # pass 1: get all the parameters in the base config
    parser.set_defaults(on_cluster=False, weighted_loss=True, reload_data=False, balance_data=False)
    cfg, _ = parser.parse_known_args()

    return cfg

def train(cfg, model_name):
    run = wandb.init(reinit=True, project=f"pp2-{model_name}")
    wandb.config.update(cfg)

    tblogger = TensorBoardLogger(save_dir=cfg.logdir)
    print(f"Logdir: {tblogger.log_dir}")

    loggers = [tblogger]

    if cfg.on_cluster:
        loggers.append(WandbLogger(save_dir=cfg.logdir))

    callbacks = [
        ModelCheckpoint(
            monitor=cfg.early_stopping_metric,
            mode="max"
        ),
        EarlyStopping(
            monitor=cfg.early_stopping_metric,
            mode="max",
            min_delta=0.01,
            patience=5
        ),
        TQDMProgressBar(
            refresh_rate=50
        )
    ]

    dataset = TMH(cfg=cfg)
    print(f"loading model {model_name}...")
    if model_name == "MLP":
        model = MLP(cfg=cfg)
    elif model_name == "CNN":
        model = CNN(cfg=cfg)
    elif model_name == "CaiT":
        model = CaiT(num_heads=1, depth=4)
    else:
        raise RuntimeError(f"Unsupported model {model_name}")
    print("loading module...")
    module = CaitModule(cfg=cfg, model=model)

    trainer = Trainer(
        logger=loggers,
        callbacks=callbacks,
        gpus=cfg.gpus if torch.cuda.is_available() else 0
    )

    print(f"start fitting {model_name} to TMH dataset...")
    trainer.fit(module, dataset, ckpt_path=None if cfg.checkpoint == "" else cfg.checkpoint)

    print("start testing...")
    trainer.test(module, dataset)

    run.finish()

def main():
    cfg = load_cfg()
    train(cfg, model_name="MLP")
    train(cfg, model_name="CNN")
    train(cfg, model_name="CaiT")

if __name__ == '__main__':
    main()
