import configparser

from model.data_module_adaptive import CoQA
from model.model import Pegasus
from trainer import build_trainer

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config/example.ini")

    model = Pegasus(config)
    data = CoQA(config, model.tokenizer)

    trainer = build_trainer(config)
    trainer.fit(model, data)
