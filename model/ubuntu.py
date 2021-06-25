import itertools
import json
import os
import re

from tqdm.contrib.concurrent import process_map
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
import pytorch_lightning as pl


class Ubuntu(pl.LightningDataModule):
    def __init__(self, hparams, tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.hparams = hparams
        self.tokenizer = tokenizer

    def prepare_data(self):
        dataset_path = os.path.join(self.hparams.dataset, "dataset_tokenized.pt")

        if os.path.isfile(dataset_path):
            print("Found tokenized dataset.")

            print(f"Loading from {datasets_path}")
            self.dataset = self.CustomDataset(
                max_input_length=self.hparams.max_input_length,
                max_output_length=self.hparams.max_output_length,
                filename=dataset_path,
            )

        else:
            print(f"Tokenizing dataset. This might take a while...")
            documents_folder = os.path.join(self.hparams.dataset, "documents")

            tokenized = self.process_and_tokenize(documents_folder)

            print(f"{len(tokenized)} samples")

            # TODO: IMPROVE HOW IT IS STORED IN MEMORY AND HOW IT IS SPLITTED
            dataset = self.CustomDataset(
                tokenized,
                tokenized,
                vocab_size=self.tokenizer.vocab_size,
            )

            dataset.save(dataset_path)
            print(f"Saved tokenized dataset to {dataset_path}")

            self.dataset = dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),
            collate_fn=AdaptiveBatch(self.tokenizer.pad_token_id),
            pin_memory=bool(torch.cuda.device_count()),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=os.cpu_count(),
            collate_fn=AdaptiveBatch(self.tokenizer.pad_token_id),
            pin_memory=bool(torch.cuda.device_count()),
        )

    def test_dataloader(self):
        return self.val_dataloader()

    def process_data(self, documents_folder):
        # separator = "<n>"
        # separator_id = self.tokenizer.encode(separator)[0]

        dataset = []

        for (dirpath, _, filenames) in os.walk(documents_folder):
            for filename in filenames:
                if ".json" in filename:
                    with open(os.path.join(dirpath, filename), "r") as f:
                        data = json.load(f)

                    data = data["contents"].replace("\n", "<n>")
                    text = re.split("<n>(?=(?:USER: )|(?:AGENT: ))", data)

                    tokenized = self.tokenizer(text)["input_ids"]
                    dataset.append(tokenized)

        return dataset


def get_int_type(vocab_size):
    if vocab_size <= 2 ** 8:
        return torch.uint8
    else:
        for int_size, int_type in {
            16: torch.int16,
            32: torch.int32,
            64: torch.int64,
        }.items():
            if vocab_size < 2 ** (int_size - 1):
                return int_type
        return torch.int64


def type_list(list_tensors, new_type):
    return [x.type(new_type) for x in list_tensors]


class AdaptiveBatch:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, samples):
        return {
            k: torch.nn.utils.rnn.pad_sequence(
                [sample[k].squeeze(0) for sample in samples],
                batch_first=True,
                padding_value=self.pad_token_id,
            )
            for k in samples[0]
        }


class CustomDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenized,
        vocab_size=None,
        filename=None,
    ):
        if filename is not None:
            self.uncompressed_type, self.input_ids, self.labels = torch.load(filename)
            self.compressed_type = self.input_ids[0].type()
            self.labels = type_list(self.labels, self.uncompressed_type)
        else:
            self.uncompressed_type = torch.long

            if vocab_size is not None:
                self.compressed_type = get_int_type(vocab_size)
            else:
                self.compressed_type = self.uncompressed_type

            self.input_ids = [
                torch.tensor(sample, dtype=self.compressed_type) for sample in src
            ]
            self.labels = [
                torch.tensor(sample, dtype=self.uncompressed_type) for sample in tgt
            ]

        if max_input_length is not None:
            self.input_ids = [
                sample[: min(len(sample), max_input_length)]
                for sample in self.input_ids
            ]
        if max_output_length is not None:
            self.labels = [
                sample[: min(len(sample), max_output_length)] for sample in self.labels
            ]

    def __getitem__(self, idx):
        item = {"input_ids": self.input_ids[idx], "labels": self.labels[idx]}
        return item

    def __len__(self):
        return len(self.input_ids)

    def save(self, filename):
        torch.save(
            (
                self.uncompressed_type,
                self.input_ids,
                type_list(self.labels, self.compressed_type),
            ),
            filename,
        )


def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


if __name__ == "__main__":
    from types import SimpleNamespace
    import yaml

    from transformers import PegasusTokenizer

    with open("config/ubuntu.yaml", "r") as f:
        hparams = yaml.full_load(f)

    hparams = SimpleNamespace(**hparams)
    tokenizer = PegasusTokenizer.from_pretrained(hparams.model_name)

    data = Ubuntu(hparams, tokenizer)
    data.prepare_data()
    data.setup()
