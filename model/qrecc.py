import hashlib
from functools import partial
from itertools import chain, cycle
import json
import multiprocessing as mp
import os
import random
import re

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import tqdm
from tqdm.contrib.concurrent import process_map
from transformers import PreTrainedTokenizer


class QReCC(pl.LightningDataModule):
    def __init__(self, hparams, tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.tokenizer = tokenizer

    def prepare_data(self):
        datasets = {
            "train": self.hparams.train_dataset,
            # "validate": self.hparams.val_dataset,
            # "test": self.hparams.test_dataset,
        }

        for mode, dataset_path in datasets.items():
            tokenized_path = self.get_tokenized_path(dataset_path)

            if os.path.isfile(tokenized_path):
                print(f"Found {dataset_path} tokenized. Loading from {tokenized_path}")
                dataset = self.CustomDataset(
                    max_input_length=self.hparams.max_input_length,
                    max_output_length=self.hparams.max_output_length,
                    filename=tokenized_path,
                )

            else:
                if os.path.isfile(dataset_path):
                    print(f"Preparing dataset. This might take a while...")

                    with open(dataset_path, "r") as f:
                        data = json.load(f)

                    data = self.build_samples_before_retrieval(data)
                    data = self.retrieve_candidates(
                        data,
                        self.hparams.passages,
                        self.hparams.max_candidates,
                        self.hparams.max_workers,
                    )
                    data = self.build_samples_after_retrieval(data)

                    dataset = self.tokenize(data)

                    print(f"{len(dataset)} conversations")

                    dataset = save_dataset(
                        dataset, dataset_path, self.tokenizer.vocab_size
                    )
                    print(f"Saved tokenized dataset to {dataset_path}")
                else:
                    print(
                        f"Dataset not found at {dataset_path}. Ignoring this dataset."
                    )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=min(os.cpu_count(), 8),
            collate_fn=AdaptiveBatch(self.tokenizer.pad_token_id),
            pin_memory=bool(torch.cuda.device_count()),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=min(os.cpu_count(), 8),
            collate_fn=AdaptiveBatch(self.tokenizer.pad_token_id),
            pin_memory=bool(torch.cuda.device_count()),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=min(os.cpu_count(), 8),
            collate_fn=AdaptiveBatch(self.tokenizer.pad_token_id),
            pin_memory=bool(torch.cuda.device_count()),
        )

    def get_tokenized_path(self, filename):
        hash_value = hash_file(filename)[:4]
        settings = "{:04d}{:04d}{:02d}{:02d}".format(
            self.hparams.max_input_length,
            self.hparams.max_output_length,
            self.hparams.max_history,
            self.hparams.max_candidates,
        )

        tokenized_path = (
            os.path.splitext(filename)[0]
            + "_tokenized_"
            + hash_value
            + "_"
            + settings
            + ".pt"
        )

        return tokenized_path

    def tokenize(self, filepaths):
        max_workers = min(self.hparams.max_workers, os.cpu_count())
        chunksize = min(1000, max(1, len(filepaths) // max_workers))

        worker_fn = partial(self.tokenize_worker, self.tokenizer)

        print("Tokenizing files")
        dataset = process_map(
            worker_fn,
            filepaths,
            max_workers=max_workers,
            chunksize=chunksize,
        )

        docids = {x.pop(0): i for i, x in enumerate(dataset)}

        return docids, dataset

    @staticmethod
    def tokenize_worker(tokenizer, filepath):
        with open(filepath, "r") as f:
            data = json.load(f)

        text = data["contents"].replace("\n", "<n>")
        text = re.split("<n>(?=(?:USER: )|(?:AGENT: ))", text)

        tokenized = tokenizer(text)["input_ids"]
        queries = ["\n".join(text[:idx]) for idx in range(1, len(text), 2)]

        # Remove eos_token from tokenized lines
        for line in tokenized:
            del line[-1]

        return [data["id"], tokenized, queries]

    @staticmethod
    def build_samples_before_retrieval(data):
        for sample in tqdm.tqdm(data, desc="Building samples before retrieval"):
            sample["Model_input"] = ("\n").join(
                sample["Context"] + [sample["Question"]]
            )

        return data

    @staticmethod
    def build_samples_after_retrieval(data):
        for sample in tqdm.tqdm(data, desc="Building samples after retrieval"):
            sample["Model_input"] = sample["Model_input"]

        return data

    @staticmethod
    def retrieve_candidates(data, passages, max_candidates=1, max_workers=16):
        from pyserini.search import SimpleSearcher

        ssearcher = SimpleSearcher(passages)
        max_workers = min(max_workers, os.cpu_count())

        queries = [sample["Model_input"] for sample in data]
        q_ids = [i for i in range(len(data))]

        print("Number of queries:", len(queries))

        chunksize = min(
            max(1000, len(queries) // 10), max(1, len(queries) // max_workers)
        )

        retrieved = []

        for i in tqdm.tqdm(
            range(0, len(queries), chunksize),
            desc=f"Retrieving candidates, {chunksize} samples at a time.",
        ):
            # Split queries in chunks
            sub_queries = queries[i : i + chunksize]
            sub_q_ids = q_ids[i : i + chunksize]

            # Batch search (by chunks)
            hits = ssearcher.batch_search(
                sub_queries, sub_q_ids, max_candidates, max_workers
            )

            import pdb

            pdb.set_trace()

            hits.sort(key=lambda hit: hit)

            for q_id in sub_q_ids:
                i = int(q_id.split("_")[0])
                candidates[i].append(
                    [
                        docids[hit.docid]
                        for hit in hits.get(q_id, [])
                        if docids[hit.docid] != i
                    ]
                )

        return data


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


# From https://www.geeksforgeeks.org/compare-two-files-using-hashing-in-python/
def hash_file(filename):
    # A arbitrary (but fixed) buffer
    # size (change accordingly)
    # 65536 = 65536 bytes = 64 kilobytes
    BUF_SIZE = 65536

    # Initializing the sha256() method
    sha256 = hashlib.sha256()

    # Opening the file provided as
    # the first commandline arguement
    with open(filename, "rb") as f:
        while True:
            # reading data = BUF_SIZE from
            # the file and saving it in a
            # variable
            data = f.read(BUF_SIZE)

            # True if eof = 1
            if not data:
                break

            # Passing that data to that sh256 hash
            # function (updating the function with
            # that data)
            sha256.update(data)

    # sha256.hexdigest() hashes all the input
    # data passed to the sha256() via sha256.update()
    # Acts as a finalize method, after which
    # all the input data gets hashed hexdigest()
    # hashes the data, and returns the output
    # in hexadecimal format
    return sha256.hexdigest()


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


def save_dataset(dataset, filename, vocab_size=None):
    if vocab_size is not None:
        compressed_type = get_int_type(vocab_size)
    else:
        compressed_type = torch.long

    for x in dataset:
        x[0] = [torch.tensor(line, dtype=compressed_type) for line in x[0]]

    torch.save(dataset, filename)

    return dataset


def load_dataset(filename):
    return torch.load(filename)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_ids,
        labels,
        vocab_size=None,
        max_input_length=None,
        max_output_length=None,
    ):
        self.uncompressed_type = torch.long

        if vocab_size is not None:
            self.compressed_type = get_int_type(vocab_size)
        else:
            self.compressed_type = self.uncompressed_type

        self.input_ids = type_list(input_ids, self.compressed_type)
        self.labels = type_list(labels, self.uncompressed_type)

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
