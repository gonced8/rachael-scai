import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger


def build_trainer(hparams):
    checkpoint_callback = ModelCheckpoint(
        filename="best", monitor="val_loss", save_last=True
    )
    lr_monitor = LearningRateMonitor()
    early_stopping = EarlyStopping(monitor="val_loss", min_delta=1e-3, patience=5)

    tb_logger = TensorBoardLogger(save_dir="checkpoints", name="")

    return pl.Trainer(
        check_val_every_n_epoch=1,
        gpus=torch.cuda.device_count(),
        default_root_dir="checkpoints",
        max_epochs=hparams.max_epochs,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        val_check_interval=hparams.val_check_interval,
        callbacks=[
            checkpoint_callback,
            lr_monitor,
            early_stopping,
        ],
        logger=tb_logger,
        log_every_n_steps=1,
        flush_logs_every_n_steps=1,
    )
