from itertools import chain
import json

import torch
from torch import nn

from datasets import load_metric
from transformers import (
    Adafactor,
    PegasusForConditionalGeneration,
    # PegasusTokenizer,
    PegasusTokenizerFast,
)

import pytorch_lightning as pl


class Pegasus(pl.LightningModule):
    def __init__(self, conf: dict):
        super().__init__()
        self.save_hyperparameters(conf)

        # self.tokenizer = PegasusTokenizer.from_pretrained(self.hparams.model_name)
        self.tokenizer = PegasusTokenizerFast.from_pretrained(self.hparams.model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(
            self.hparams.model_name
        )
        self.rouge_metric = load_metric("rouge", cache_dir=".cache/")

        # self.freeze_embeds()

    def freeze_params(self, model: nn.Module):
        """Set requires_grad=False for each of model.parameters()"""
        for par in model.parameters():
            par.requires_grad = False

    def freeze_embeds(self):
        """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
        self.freeze_params(self.model.model.shared)
        for d in [self.model.model.encoder, self.model.model.decoder]:
            self.freeze_params(d.embed_positions)
            self.freeze_params(d.embed_tokens)

    def forward(self, input_ids, attention_mask=None, labels=None):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        return output

    def training_step(self, batch, batch_idx):
        output = self.forward(
            batch["input_ids"], batch["attention_mask"], batch["labels"]
        )
        loss = output.loss

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.forward(
            batch["input_ids"], batch["attention_mask"], batch["labels"]
        )
        loss = output.loss

        predictions = self.tokenizer.batch_decode(
            torch.argmax(output.logits, dim=2),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        references = self.tokenizer.batch_decode(
            batch["labels"], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        self.rouge_metric.add_batch(predictions=predictions, references=references)

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def validation_epoch_end(self, outputs):
        rouge_score = self.rouge_metric.compute()
        rouge_score = parse_rouge_score(rouge_score)

        self.log_dict(rouge_score, prog_bar=True)
        return

    def on_epoch_start(self):
        print()

    def test_step(self, batch, batch_idx):
        output = self.model.generate(
            batch["input_ids"],
            max_length=self.hparams.max_output_length,
            do_sample=True,
        )

        predictions = self.tokenizer.batch_decode(
            output,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        if "labels" in batch:
            references = self.tokenizer.batch_decode(
                batch["labels"],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            self.rouge_metric.add_batch(predictions=predictions, references=references)

        return [
            {
                "Conversation_no": batch["Conversation_no"][i],
                "Turn_no": batch["Turn_no"][i],
                "Model_passages": batch["Model_passages"][i],
                "Model_answer": predictions[i],
            }
            for i in range(len(predictions))
        ]

    def test_epoch_end(self, outputs):
        # Merge outputs of the multiple steps
        outputs = list(chain(*outputs))

        # Save output
        filename = "run.json"
        with open(filename, "w") as f:
            json.dump(outputs, f, indent=2)
            print(f"Saved test output to: {filename}")

    def generate(self, input_ids, max_length=None, *args, **kargs):
        if max_length is None:
            max_length = self.hparams.max_output_length

        if not batch.is_cuda:
            batch.to(self.device)

        output = self.model.generate(
            input_ids,
            max_length,
            *args,
            **kargs,
        )

        predictions = self.tokenizer.batch_decode(
            output,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        return predictions

    def configure_optimizers(self):
        self.lr = self.hparams.learning_rate
        flag = self.lr == "None"
        self.lr = float(self.lr) if not flag else None

        optimizer = Adafactor(
            self.parameters(),
            lr=self.lr,
            scale_parameter=flag,
            relative_step=flag,
            warmup_init=flag,
        )

        # scheduler = {
        #    "scheduler": AdafactorSchedule(optimizer),
        #    "interval": "step",
        # }

        # return [optimizer], [scheduler]
        return optimizer


def parse_rouge_score(result):
    return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}
