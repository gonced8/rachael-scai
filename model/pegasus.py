import torch
from torch import nn

from datasets import load_metric
from transformers import (
    Adafactor,
    PegasusForConditionalGeneration,
    PegasusTokenizer,
    # PegasusTokenizerFast,
)

import pytorch_lightning as pl


class Pegasus(pl.LightningModule):
    def __init__(self, conf: dict):
        super().__init__()
        self.save_hyperparameters(conf)

        self.tokenizer = PegasusTokenizer.from_pretrained(self.hparams.model_name)
        # self.tokenizer = PegasusTokenizerFast.from_pretrained(self.hparams.model_name)
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
        if False:
            print(
                "Input:",
                self.tokenizer.decode(input_ids[0]).replace("<n>", "\n"),
                separator="\n",
            )
            print(
                "Target:",
                self.tokenizer.decode(labels[0]).replace("<n>", "\n"),
                separator="\n",
            )
            print()

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        return output

    def training_step(self, batch, batch_idx):
        output = self.forward(**batch)
        loss = output.loss

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.forward(**batch)
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
        references = self.tokenizer.batch_decode(
            batch["labels"], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        self.rouge_metric.add_batch(predictions=predictions, references=references)
        rouge_score = self.rouge_metric.compute()
        rouge_score = parse_rouge_score(rouge_score)

        print(rouge_score)
        return

    def generate(self, batch):
        print(batch.is_cuda, self.model.is_cuda)
        batch.to(self.device)
        output = self.model.generate(
            **kargs,
            max_length=self.hparams.max_output_length,
            do_sample=True,
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
