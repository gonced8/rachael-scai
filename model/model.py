import torch

from datasets import load_metric
from transformers import (
    AdaFactor,
    PegasusForConditionalGeneration,
    PegasusTokenizer,
)

import pytorch_lightning as pl


class Pegasus(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.tokenizer = PegasusTokenizer.from_pretrained(
            self.config["Model"]["model_name"]
        )
        self.model = PegasusForConditionalGeneration.from_pretrained(
            self.config["Model"]["model_name"]
        )
        self.rouge_metric = load_metric("rouge")

    def forward(
        self,
        input_ids,
        decoder_input_ids,
        attention_mask=None,
        decoder_attention_mask=None,
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            return_dict=True,
        )

    def training_step(self, batch, batch_idx):
        output = self.forward(**batch)
        loss = output.loss

        return loss

    def validation_step(self, batch, batch_idx):
        output = self.forward(**batch)
        loss = output.loss

        self.rouge_metric.add_batch(
            predictions=output, references=batch["decoder_input_ids"]
        )
        rouge_score = self.rouge_metric.compute()

        self.log("val_loss", loss, prog_bar=True)
        self.log("rouge", rouge_score, prog_bar=True)
        return {"val_loss": loss, "rouge": rouge_score}

    # def test_step(self, batch, batch_idx):
    #    return

    def configure_optimizers(self):
        self.lr = self.config["Model"]["learning_rate"]
        flag = self.lr == "None"
        self.lr = float(self.lr) if not flag else None

        optimizer = AdaFactor(
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
