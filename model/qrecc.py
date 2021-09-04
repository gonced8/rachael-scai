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
    def __init__(self, conf, tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.hparams.update(conf)
        self.tokenizer = tokenizer

    def prepare_data(self):
        datasets = self.get_datasets_paths()

        for mode, dataset_path in datasets.items():
            # Work with list of filenames
            if isinstance(dataset_path, str):
                dataset_path = [dataset_path]

            tokenized_path = self.get_tokenized_path(mode, dataset_path)

            if os.path.isfile(tokenized_path):
                print(f"Found {mode} dataset tokenized. Loading from {tokenized_path}")
                dataset = CustomDataset(
                    filename=tokenized_path,
                )

            else:
                retrieved_path = self.get_retrieved_path(mode, dataset_path)

                # Retrieval model
                from pyserini.search import SimpleSearcher

                ssearcher = SimpleSearcher(
                    os.path.join(self.hparams.input_dir, self.hparams.passages)
                )

                # Either load dataset with retrieved passages
                if os.path.isfile(retrieved_path):
                    print(f"Loading dataset from {retrieved_path}...")
                    with open(retrieved_path, "r") as f:
                        data = json.load(f)

                # Or load dataset and retrieve them
                elif all(os.path.isfile(path) for path in dataset_path):
                    print(f"Preparing dataset. This might take a while...")

                    # Read dataset_path files and merge into a single data "JSON"
                    data = None
                    for path in dataset_path:
                        with open(path, "r") as f:
                            file_data = json.load(f)

                        if data is None:
                            data = file_data
                        else:
                            for sample1, sample2 in zip(data, file_data):
                                if (
                                    sample1["Conversation_no"]
                                    == sample2["Conversation_no"]
                                    and sample1["Turn_no"] == sample2["Turn_no"]
                                ):
                                    sample1.update(sample2)
                                else:
                                    print(f"Datasets of {mode} are inconsistent.")

                    # Build samples to be used for retrieval
                    data = self.build_samples_before_retrieval(data)

                    # Retrieve relavant passages
                    data = self.retrieve_candidates(
                        data,
                        ssearcher,
                        self.hparams.max_candidates,
                        self.hparams.max_workers,
                    )

                    # Save dataset with retrieved passages
                    if self.hparams.cache_dataset:
                        with open(retrieved_path, "w") as f:
                            json.dump(data, f)
                        print(
                            f"Saved dataset with retrieved passages in {retrieved_path}"
                        )

                else:
                    print(
                        f"Dataset not found at {dataset_path}. Ignoring this dataset."
                    )
                    continue

                # Build samples considering retrieved passages
                data = self.build_samples_after_retrieval(data, ssearcher)

                # Tokenize

                # tokenized = self.tokenize_parallel(data)
                tokenized = self.tokenize(data)

                dataset = CustomDataset(
                    **tokenized, vocab_size=self.tokenizer.vocab_size
                )

                print(f"{len(dataset)} samples")

                # Save tokenized dataset
                if self.hparams.cache_dataset:
                    dataset.save(tokenized_path)
                    print(f"Saved tokenized dataset to {tokenized_path}")

            if mode == "train":
                self.train_dataset = dataset
            elif mode == "validate":
                self.val_dataset = dataset
            elif mode == "test":
                self.test_dataset = dataset
            else:
                print("Unrecognized mode. Only supports train, validate, and test.")

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

    def get_datasets_paths(self):
        datasets = {}

        if self.hparams.train_dataset:
            datasets["train"] = os.path.join(
                self.hparams.input_dir, self.hparams.train_dataset
            )
        if self.hparams.val_dataset:
            datasets["validate"] = os.path.join(
                self.hparams.input_dir, self.hparams.val_dataset
            )
        if self.hparams.test_dataset:
            datasets["test"] = os.path.join(
                self.hparams.input_dir, self.hparams.test_dataset
            )

        return datasets

    def get_tokenized_path(self, mode, filenames):
        hash_value = hash_file(filenames)[:4]
        settings = "{:04d}{:04d}{:02d}{:02d}".format(
            self.hparams.max_input_length,
            self.hparams.max_output_length,
            self.hparams.max_history,
            self.hparams.max_candidates,
        )

        tokenized_path = (
            os.path.join(os.path.dirname(filenames[0]), mode)
            + "_tokenized_"
            + hash_value
            + "_"
            + settings
            + ".pt"
        )

        return tokenized_path

    def get_retrieved_path(self, mode, filenames):
        hash_value = hash_file(filenames)[:4]
        settings = "{:02d}{:02d}".format(
            self.hparams.max_history,
            self.hparams.max_candidates,
        )

        retrieved_path = (
            os.path.join(os.path.dirname(filenames[0]), mode)
            + "_retrieved_"
            + hash_value
            + "_"
            + settings
            + ".pt"
        )

        return retrieved_path

    def tokenize(self, data):
        src = [sample["Model_input"] for sample in data]
        if "Truth_answer" in data[0]:
            tgt = [sample["Truth_answer"] for sample in data]
        else:
            tgt = None

        if "pegasus" in self.hparams.model_name:
            src = [sample.replace("\n", "<n>") for sample in src]
            if tgt is not None:
                tgt = [sample.replace("\n", "<n>") for sample in tgt]

        dataset = {
            "Conversation_no": [sample["Conversation_no"] for sample in data],
            "Turn_no": [sample["Turn_no"] for sample in data],
            "Model_passages": [sample["Model_passages"] for sample in data],
        }

        src_tokenized = []
        for sequence in tqdm.tqdm(src, desc="Tokenizing source..."):
            src_tokenized.append(
                self.tokenizer.encode(
                    sequence,
                    truncation=True,
                    max_length=self.hparams.max_input_length,
                    return_tensors="pt",
                )[0]
            )
        dataset["input_ids"] = src_tokenized

        if tgt is not None:
            tgt_tokenized = []
            for sequence in tqdm.tqdm(tgt, desc="Tokenizing target..."):
                tgt_tokenized.append(
                    self.tokenizer.encode(
                        sequence,
                        truncation=True,
                        max_length=self.hparams.max_output_length,
                        return_tensors="pt",
                    )[0]
                )
            dataset["labels"] = tgt_tokenized

        return dataset

    def tokenize_parallel(self, data):
        max_workers = min(self.hparams.max_workers, os.cpu_count())
        chunksize = min(1000, max(1, len(data) // max_workers))

        worker_fn = partial(
            self.tokenize_worker,
            self.tokenizer,
            self.hparams.max_input_length,
            self.hparams.max_output_length,
        )

        print("Tokenizing samples")
        data_tokenized = process_map(
            worker_fn,
            data,
            max_workers=max_workers,
            chunksize=chunksize,
        )

        # Convert to dict
        dataset = {
            "Conversation_no": [sample["Conversation_no"] for sample in data],
            "Turn_no": [sample["Turn_no"] for sample in data],
            "Model_passages": [sample["Model_passages"] for sample in data],
        }

        # Check if dataset has labels
        if isinstance(dataset[0], list):
            # Transpose dataset
            data_tokenized = [list(i) for i in zip(*data_tokenized)]
            dataset["input_ids"] = data_tokenized[0]
            dataset["labels"] = data_tokenized[1]
        else:
            dataset["input_ids"] = dataset_tokenized

        return dataset

    @staticmethod
    def tokenize_worker(tokenizer, max_input_length, max_output_length, sample):
        src = sample["Model_input"]
        if "Truth_answer" in sample:
            tgt = sample["Truth_answer"]
        else:
            tgt = None

        if "pegasus" in tokenizer.name_or_path:
            src = src.replace("\n", "<n>")
            if tgt is not None:
                tgt = tgt.replace("\n", "<n>")

        src_tokenized = tokenizer.encode(
            src, truncation=True, max_length=max_input_length, return_tensors="pt"
        )[0]

        if tgt is not None:
            tgt_tokenized = tokenizer.encode(
                tgt, truncation=True, max_length=max_output_length, return_tensors="pt"
            )[0]

            return [src_tokenized, tgt_tokenized]
        else:
            return src_tokenized

    @staticmethod
    def build_samples_before_retrieval(data):
        context_exists = "Context" in data[0]

        conversation = None

        for sample in tqdm.tqdm(data, desc="Building samples before retrieval"):
            if context_exists:
                sample["Model_input"] = ("\n").join(
                    sample["Context"] + [sample["Question"]]
                )
            else:
                if conversation != sample["Conversation_no"]:
                    question = []

                question.append(sample["Question"])
                sample["Model_input"] = "\n".join(question)

        return data

    @staticmethod
    def build_samples_after_retrieval(data, ssearcher):
        for sample in tqdm.tqdm(data, desc="Building samples after retrieval"):
            docs = [ssearcher.doc(docid) for docid in sample["Model_passages"]]
            passages = [json.loads(doc.raw())["contents"] for doc in docs]
            sample["Model_input"] = "\n\n".join([sample["Model_input"]] + passages)

        return data

    @staticmethod
    def retrieve_candidates(data, ssearcher, max_candidates=1, max_workers=16):
        max_workers = min(max_workers, os.cpu_count())

        queries = [sample["Model_input"] for sample in data]
        q_ids = [str(i) for i in range(len(data))]

        print("Number of queries:", len(queries))

        chunksize = min(
            max(1000, len(queries) // 10), max(1, len(queries) // max_workers)
        )

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

            # Update data with retrieved passages
            for q_id, candidates in hits.items():
                data[int(q_id)]["Model_passages"] = {
                    hit.docid: hit.score for hit in candidates
                }

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
    if isinstance(list_tensors, list):
        return [x.type(new_type) for x in list_tensors]
    else:
        print("Using function type_list in a object that is not a list. Ignoring...")
        return list_tensors


# From https://www.geeksforgeeks.org/compare-two-files-using-hashing-in-python/
def hash_file(filenames):
    # A arbitrary (but fixed) buffer
    # size (change accordingly)
    # 65536 = 65536 bytes = 64 kilobytes
    BUF_SIZE = 65536

    # Initializing the sha256() method
    sha256 = hashlib.sha256()

    # Opening the file provided as
    # the first commandline arguement
    for filename in filenames:
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
        batch = {
            "Conversation_no": [sample["Conversation_no"] for sample in samples],
            "Turn_no": [sample["Turn_no"] for sample in samples],
            "Model_passages": [sample["Model_passages"] for sample in samples],
        }

        batch["input_ids"] = torch.nn.utils.rnn.pad_sequence(
            [sample["input_ids"] for sample in samples],
            batch_first=True,
            padding_value=self.pad_token_id,
        )

        batch["attention_mask"] = torch.nn.utils.rnn.pad_sequence(
            [
                torch.ones_like(sample["input_ids"], dtype=torch.bool)
                for sample in samples
            ],
            batch_first=True,
        )

        if "labels" in samples[0]:
            batch["labels"] = torch.nn.utils.rnn.pad_sequence(
                [sample["labels"] for sample in samples],
                batch_first=True,
                padding_value=self.pad_token_id,
            )

        return batch


class CustomDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        Conversation_no=None,
        Turn_no=None,
        Model_passages=None,
        input_ids=None,
        labels=None,
        vocab_size=None,
        filename=None,
        **kargs,
    ):
        if filename is not None:
            (
                self.uncompressed_type,
                self.Conversation_no,
                self.Turn_no,
                self.Model_passages,
                self.input_ids,
                self.labels,
            ) = torch.load(filename)
            self.compressed_type = self.input_ids[0].type()
            self.labels = type_list(self.labels, self.uncompressed_type)
        else:
            self.uncompressed_type = torch.long

            if vocab_size is not None:
                self.compressed_type = get_int_type(vocab_size)
            else:
                self.compressed_type = self.uncompressed_type

            self.Conversation_no = Conversation_no
            self.Turn_no = Turn_no
            self.Model_passages = Model_passages
            self.input_ids = type_list(input_ids, self.compressed_type)
            self.labels = type_list(labels, self.uncompressed_type)

    def __getitem__(self, idx):
        item = {
            "Conversation_no": self.Conversation_no[idx],
            "Turn_no": self.Turn_no[idx],
            "Model_passages": self.Model_passages[idx],
            "input_ids": self.input_ids[idx],
        }

        if self.labels is not None:
            item["labels"] = self.labels[idx]

        return item

    def __len__(self):
        return len(self.input_ids)

    def save(self, filename):
        torch.save(
            (
                self.uncompressed_type,
                self.Conversation_no,
                self.Turn_no,
                self.Model_passages,
                self.input_ids,
                type_list(self.labels, self.compressed_type)
                if self.labels is not None
                else None,
            ),
            filename,
        )


if __name__ == "__main__":
    from types import SimpleNamespace
    import yaml

    from transformers import PegasusTokenizer

    with open("config/ubuntu.yaml", "r") as f:
        hparams = yaml.full_load(f)

    hparams = SimpleNamespace(**hparams)
    tokenizer = PegasusTokenizer.from_pretrained(hparams.model_name)

    data = QReCC(hparams, tokenizer)
    data.prepare_data()
    data.setup()
