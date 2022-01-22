from pathlib import Path

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
    mode_group.add_argument("--on_polyaxon", action="store_true")
    mode_group.add_argument("--logdir", type=str, default="logs")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TRAINING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    training_group = parser.add_argument_group(title='Training options')
    training_group.add_argument("--gpus", type=int, default=-1)
    training_group.add_argument("--early_stopping_metric", type=str, default="val_acc")
    training_group.add_argument("--optimizer", type=str, default='Adam')
    training_group.add_argument("--learning_rate", type=float, default=1e-5)
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
    dataloader_group.add_argument("--no-mean_embedding", dest="mean_embedding", action="store_false")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MODEL ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    model_group = parser.add_argument_group(title='Model options')
    model_group.add_argument("--model", type=str, default="CNN")
    model_group.add_argument("--input_dim", type=int, default=25)
    model_group.add_argument("--hidden_dims_list", type=str, default="20,10")
    model_group.add_argument("--dropout_p", type=float, default=0.3)
    model_group.add_argument("--embed_dim", type=int, default=1024)
    model_group.add_argument("--num_heads", type=int, default=8)
    model_group.add_argument("--depth", type=int, default=24)
    model_group.add_argument("--depth_token_only", type=int, default=2)
    model_group.add_argument("--mlp_ratio", type=float, default=4.)
    model_group.add_argument("--mlp_ratio_token_only", type=float, default=4.0)
    model_group.add_argument("--drop_rate", type=float, default=0.)
    model_group.add_argument("--attn_drop_rate", type=float, default=0.)
    model_group.add_argument("--drop_path_rate", type=float, default=0.)
    model_group.add_argument("--init_scale", type=float, default=1e-5)
    model_group.add_argument("--no-qkv_bias", dest="qkv_bias", action="store_false")
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
    parser.set_defaults(
        on_cluster=False,
        on_polyaxon=False,
        weighted_loss=True,
        reload_data=False,
        balance_data=False,
        mean_embedding=True,
        qkv_bias=True
    )
    cfg, _ = parser.parse_known_args()

    return cfg

def train(cfg, model_name):
    loggers = []
    if cfg.on_polyaxon:
        from common.plx_logger import PolyaxonLogger
        poly_logger = PolyaxonLogger(cfg)
        loggers.append(poly_logger)
        cfg.logdir = str(poly_logger.output_path / poly_logger.name / f'version_{poly_logger.version}')
    if cfg.on_cluster:
        run = wandb.init(reinit=True, project=f"pp2-{model_name}")
        wandb.config.update(cfg)
        loggers.append(WandbLogger(save_dir=cfg.logdir))
    loggers.append(TensorBoardLogger(save_dir=cfg.logdir))
    print(f"Logdir: {cfg.logdir}")

    ckpt_dir = Path(cfg.logdir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
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
    elif "CaiT" in model_name:
        model = CaiT(cfg=cfg)
    else:
        raise RuntimeError(f"Unsupported model {model_name}")
    print("loading module...")
    module = CaitModule(cfg=cfg, model=model)

    trainer = Trainer(
        logger=loggers,
        callbacks=callbacks,
        gpus=cfg.gpus if torch.cuda.is_available() else 0,
    )

    print(f"start fitting {model_name} to TMH dataset...")
    trainer.fit(module, dataset, ckpt_path=None if cfg.checkpoint == "" else cfg.checkpoint)

    print("start testing...")
    trainer.test(module, dataset)

    if cfg.on_cluster:
        run.finish()

def main():
    cfg = load_cfg()
    train(cfg, model_name="MLP")
    train(cfg, model_name="CNN")

    # can't currently train transformers with minibatches
    cfg.batch_size = 1
    cfg.mean_embedding = False

    cfg.num_heads = 1
    cfg.depth = 4
    cfg.depth_token_only = 1
    train(cfg, model_name="CaiT-XS")

    cfg.num_heads = 2
    cfg.depth = 12
    cfg.depth_token_only = 1
    train(cfg, model_name="CaiT-S")

    cfg.num_heads=4
    cfg.depth=24
    cfg.depth_token_only=2
    train(cfg, model_name="CaiT-M")

    cfg.num_heads = 8
    cfg.depth = 24
    cfg.depth_token_only = 2
    train(cfg, model_name="CaiT-L")

if __name__ == '__main__':
    main()
