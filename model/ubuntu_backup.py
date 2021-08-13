from collections import defaultdict, ChainMap
from functools import partial
from itertools import chain, cycle
import json
import multiprocessing as mp
import os
import random
import re

from pyserini.search import SimpleSearcher
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import tqdm
from tqdm.contrib.concurrent import process_map
from transformers import PreTrainedTokenizer


class Ubuntu(pl.LightningDataModule):
    def __init__(self, hparams, tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.hparams = hparams
        self.tokenizer = tokenizer

    def prepare_data(self):
        dataset_path = os.path.join(self.hparams.dataset, "dataset_tokenized.pt")

        if os.path.isfile(dataset_path):
            print("Found tokenized dataset.")

            print(f"Loading from {dataset_path}")
            dataset = load_dataset(dataset_path)

        else:
            print(f"Preparing dataset. This might take a while...")
            self.documents_folder = os.path.join(self.hparams.dataset, "documents")
            filepaths = [doc.path for doc in os.scandir(self.documents_folder)][:2000]

            dataset = self.tokenize(filepaths)
            dataset = self.retrieve_candidates(dataset)
            print(f"{len(dataset)} conversations")

            dataset = save_dataset(dataset, dataset_path, self.tokenizer.vocab_size)
            print(f"Saved tokenized dataset to {dataset_path}")

        dataset = self.generate_samples(dataset)

        idx = round(len(dataset["input_ids"]) * self.hparams.split)

        self.train_dataset = CustomDataset(
            dataset["input_ids"][:idx],
            dataset["labels"][:idx],
            vocab_size=self.tokenizer.vocab_size,
            max_input_length=self.hparams.max_input_length,
            max_output_length=self.hparams.max_output_length,
        )

        self.val_dataset = CustomDataset(
            dataset["input_ids"][idx:],
            dataset["labels"][idx:],
            vocab_size=self.tokenizer.vocab_size,
            max_input_length=self.hparams.max_input_length,
            max_output_length=self.hparams.max_output_length,
        )

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

        return dict(ChainMap(*dataset))

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

        return {
            data["id"]: {
                "tokenized": tokenized,
                "queries": queries,
                "candidates": [],
            }
        }

    def retrieve_candidates(self, dataset):
        ssearcher = SimpleSearcher("data/ubuntu/index/sparse")

        q_ids = [
            docid + "-" + str(i)
            for docid, interaction in dataset.items()
            for i in range(len(interaction["queries"]))
        ]
        queries = list(
            chain(*[interaction.pop("queries") for _, interaction in dataset.items()])
        )

        """
        print("Number of queries:", len(queries))

        hits = ssearcher.batch_search(
            queries, q_ids, self.hparams.max_candidates, max_workers
        )

        for q_id in q_ids:
            docid = q_id.split("-")[0]
            candidates[docid].append(
                [hit.docid for hit in hits[q_id] if hit.docid != docid]
            )
        """

        max_workers = min(self.hparams.max_workers, os.cpu_count())
        chunksize = min(
            max(1000, len(queries) // 100), max(1, len(queries) // max_workers)
        )

        for i in tqdm.tqdm(
            range(0, len(queries), chunksize), desc="Retrieving candidates"
        ):
            sub_queries = queries[i : i + chunksize]
            sub_q_ids = q_ids[i : i + chunksize]

            hits = ssearcher.batch_search(
                sub_queries, sub_q_ids, self.hparams.max_candidates, max_workers
            )

            for q_id in sub_q_ids:
                docid = q_id.split("-")[0]
                dataset[docid]["candidates"].append(
                    [hit.docid for hit in hits.get(q_id, []) if hit.docid != docid]
                )

        return dataset

    def generate_samples(self, dataset):
        dtype = next(iter(dataset.values()))["tokenized"][0][0].dtype

        separator = "<n>"
        separator_id = torch.tensor(
            [self.tokenizer.convert_tokens_to_ids(separator)], dtype=dtype
        )
        eos_token_id = torch.tensor([self.tokenizer.eos_token_id], dtype=dtype)

        max_workers = min(self.hparams.max_workers, os.cpu_count())
        chunksize = min(1000, max(1, len(dataset) // max_workers))

        worker_fn = partial(
            self.generate_samples_worker,
            self.hparams.max_input_length,
            self.hparams.max_output_length,
            separator_id,
            eos_token_id,
            dataset,
        )

        print("Generating samples")
        dataset = process_map(
            worker_fn,
            dataset.values(),
            max_workers=max_workers,
            chunksize=chunksize,
        )

        return {k: list(chain(*[sample[k] for sample in dataset])) for k in dataset[0]}

    def test(self, interaction):
        print(interaction)

    @staticmethod
    def generate_samples_worker(
        max_input_length,
        max_output_length,
        separator_id,
        eos_token_id,
        dataset,
        interaction,
    ):
        print("AQUI")
        input_ids = []
        labels = []

        for i, idx in enumerate(range(1, len(interaction["tokenized"]), 2)):
            # Select question with appropriate history
            question = interaction["tokenized"][:idx]
            # Separate lines using newline token
            question = list(chain(*zip(question, cycle([separator_id]))))

            if len(question) > hparams.max_input_length:
                break

            n_candidates = random.randint(0, len(interaction["candidates"][i]) - 1)
            retrieved = [
                dataset[other]["tokenized"] for other in interaction["candidates"][i]
            ]

            for candidate in retrieved:
                if (
                    len(question) + len(candidate) + 2  # 1 <n> token + 1 eos_token
                    > max_input_length
                ):
                    break

                question.append(separator_id)
                question.extend(candidate)
                question.append(separator_id)

            question[-1] = eos_token_id
            question = torch.cat(question)

            answer = interaction["tokenized"][idx]
            if len(answer) >= max_output_length:
                answer = answer[: max_output_length - 1]
            answer = torch.cat([answer, eos_token_id])

            input_ids.append(question)
            labels.append(answer)

        return {"input_ids": input_ids, "labels": labels}


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


def save_dataset(dataset, filename, vocab_size=None):
    if vocab_size is not None:
        compressed_type = get_int_type(vocab_size)
    else:
        compressed_type = torch.long

    for docid, interaction in dataset.items():
        dataset[docid]["tokenized"] = [
            torch.tensor(line, dtype=compressed_type)
            for line in interaction["tokenized"]
        ]

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


def type_list(list_tensors, new_type):
    return [x.type(new_type) for x in list_tensors]


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
