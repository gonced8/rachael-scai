from itertools import chain
import json

#from datasets import load_metric
from transformers import T5ForConditionalGeneration, T5TokenizerFast

import pytorch_lightning as pl


class T5(pl.LightningModule):
    def __init__(self, conf: dict):
        super().__init__()
        self.save_hyperparameters(conf)

        self.tokenizer = T5TokenizerFast.from_pretrained(self.hparams.rewrite_model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.hparams.rewrite_model_name)
        #self.rouge_metric = load_metric("rouge", cache_dir=".cache/")

    def forward(self, input_ids, attention_mask=None, labels=None):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        return output

    def test_step(self, batch, batch_idx):
        output = self.model.generate(
            batch["input_ids"],
            max_length=self.hparams.rewrite_max_output_length,
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

            #self.rouge_metric.add_batch(predictions=predictions, references=references)

        return [
            {
                "Conversation_no": batch["Conversation_no"][i],
                "Turn_no": batch["Turn_no"][i],
                "Model_rewrite": predictions[i],
            }
            for i in range(len(predictions))
        ]

    def test_epoch_end(self, outputs):
        # Merge outputs of the multiple steps
        outputs = list(chain(*outputs))

        # Save output
        filename = "rewrite.json"
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
