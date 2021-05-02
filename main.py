import configparser

import torch

from transformers import (
    PegasusForConditionalGeneration,
    PegasusTokenizer,
)

from model.data_module import CoQA
from model.model import Pegasus

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config/example.ini")

    checkpoint_callback = ModelCheckpoint(
        filename="best", monitor="val_loss", save_last=True
    )
    lr_monitor = LearningRateMonitor()
    early_stopping = EarlyStopping(monitor="val_loss", min_delta=1e-3, patience=5)

    trainer = pl.Trainer(
        check_val_every_n_epoch=1,
        gpus=torch.cuda.device_count(),
        default_root_dir="checkpoints",
        max_epochs=int(config["Model"]["max_epochs"]),
        callbacks=[
            checkpoint_callback,
            lr_monitor,
            early_stopping,
        ],
    )

    model = Pegasus(config)
    data = CoQA(config, model.tokenizer)

    train.fit(model, data)
