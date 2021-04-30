import configparser

import torch

from transformers import (
    PegasusForConditionalGeneration,
    PegasusTokenizer,
    Trainer,
)

from model.data_module import DataModule

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config/example.ini")
    tokenizer = PegasusTokenizer.from_pretrained(config["Model"]["pretrained_model"])
    data = DataModule(config, tokenizer)
    data.prepare_data()
