import torch

from datasets import load_metric
from transformers import (
    Adafactor,
    PegasusForConditionalGeneration,
    PegasusTokenizer,
)

import pytorch_lightning as pl


class Pegasus(pl.LightningModule):
    def __init__(self, hparams: dict):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.tokenizer = PegasusTokenizer.from_pretrained(
            self.hparams.model_name, cache_dir=".cache/"
        )
        self.model = PegasusForConditionalGeneration.from_pretrained(
            self.hparams.model_name, cache_dir=".cache/"
        )
        self.rouge_metric = load_metric("rouge", cache_dir=".cache/")

    def forward(self, input_ids, labels=None):
        output = self.model(
            input_ids=input_ids,
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
        #    "scheduler": OneCycleLR(
        #        optimizer,
        #        max_lr=1e-4,
        #        total_steps=30,
        #        div_factor=10,
        #        final_div_factor=10,
        #        verbose=True,
        #    ),
        #    "interval": "epoch",
        # }
        # return [optimizer], [scheduler]
        return optimizer


def parse_rouge_score(result):
    return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}
