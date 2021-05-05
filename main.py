import yaml

from pytorch_lightning.utilities.cli import LightningCLI

from model.data_module_adaptive import CoQA
from model.model import Pegasus
from trainer import build_trainer

if __name__ == "__main__":
    cli = LightningCLI(Pegasus, CoQA)

    # model = Pegasus(config)
    # data = CoQA(config, model.tokenizer)

    # trainer = build_trainer(config)
    # trainer.fit(model, data)
    # trainer.save_checkpoint("example.ckpt")
