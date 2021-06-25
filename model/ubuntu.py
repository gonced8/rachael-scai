import itertools
import json
import os
import re

# from tqdm.contrib.concurrent import process_map
from pyserini.search import SimpleSearcher
import torch
from torch.utils.data import DataLoader
import tqdm
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

            print(f"Loading from {dataset_path}")
            dataset = load_dataset(dataset_path)

        else:
            print(f"Tokenizing dataset. This might take a while...")
            documents_folder = os.path.join(self.hparams.dataset, "documents")

            dataset = self.process_and_tokenize(documents_folder)
            print(f"{len(dataset['docid'])} conversations")

            dataset = save_dataset(dataset, dataset_path)
            print(f"Saved tokenized dataset to {dataset_path}")

        dataset = self.generate_samples(**dataset)

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

    def process_and_tokenize(self, documents_folder):
        dataset = {"docid": [], "tokenized": [], "candidates": []}

        for (dirpath, _, filenames) in os.walk(documents_folder):
            for filename in filenames:
                if ".json" in filename:
                    with open(os.path.join(dirpath, filename), "r") as f:
                        data = json.load(f)

                    text = data["contents"].replace("\n", "<n>")
                    text = re.split("<n>(?=(?:USER: )|(?:AGENT: ))", text)

                    tokenized = self.tokenizer(text)["input_ids"]

                    # Remove eos_token from tokenized lines
                    for line in tokenized:
                        del line[-1]

                    # Retrieve candidates
                    for idx in range(1, len(text), 2):
                        question = "\n".join(text[:idx])
                        hits = ssearcher.search(question)[: self.hparams.max_candidates]
                        import pdb

                        pdb.set_trace()
                        # candidates = ssearcher.search(question)[: self.hparams.max_candidates]

                    dataset["docid"].append(data["id"])
                    dataset["tokenized"].append(tokenized)

        return dataset

    def generate_samples(self, docid, tokenized):
        input_ids = []
        labels = []
        candidates = []

        separator = "<n>"
        separator_id = torch.tensor(
            [self.tokenizer.convert_tokens_to_ids(separator)],
            dtype=tokenized[0][0].dtype,
        )
        separator_id = itertools.cycle([separator_id])

        eos_token_id = torch.tensor(
            [self.tokenizer.eos_token_id], dtype=tokenized[0][0].dtype
        )

        ssearcher = SimpleSearcher("data/ubuntu/index/sparse")

        for i, interaction in tqdm.tqdm(
            zip(docid, tokenized), desc="Generating samples"
        ):
            for idx in range(1, len(interaction), 2):
                # Select question with appropriate history
                question = interaction[:idx]
                # Separate lines using newline token
                question = list(itertools.chain(*zip(question, separator_id)))
                question[-1] = eos_token_id
                question = torch.cat(question)
                # TODO: ADD RETRIEVED CANDIDATES

                answer = interaction[idx]
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

    dataset["tokenized"] = [
        [torch.tensor(line, dtype=compressed_type) for line in interaction]
        for interaction in dataset["tokenized"]
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
