import torch
from dataset import CoqaDataset
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

model_name = "google/pegasus-xsum"
model = PegasusForConditionalGeneration.from_pretrained(model_name)
tokenizer = PegasusTokenizer.from_pretrained(model_name)
train_filename = "dataset/train.json"
eval_filename = "dataset/test.json"

training_args = TrainingArguments(
    output_dir="results",  # output directory
    num_train_epochs=3,  # total # of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,  # batch size for evaluation
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    adafactor=True,  # use Adafactor instead of AdamW
    logging_dir="logs",  # directory for storing logs
)

trainer = Trainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=CoqaDataset(train_filename),  # training dataset
    eval_dataset=CoqaDataset(eval_filename),  # evaluation dataset
)

trainer.train()
trainer.evaluate()
