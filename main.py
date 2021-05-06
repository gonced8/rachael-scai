import yaml

from model.data_module import CoQA
from model.model import Pegasus
from trainer import build_trainer

if __name__ == "__main__":
    with open("config/example.yaml", "r") as f:
        hparams = yaml.full_load(f)

    model = Pegasus(hparams)
    data = CoQA(model.hparams, model.tokenizer)

    trainer = build_trainer(model.hparams)
    trainer.fit(model, data)
