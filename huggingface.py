from model.model import Pegasus

model = Pegasus.load_from_checkpoint(
    "checkpoints/version_0/checkpoints/best.ckpt",
    hparams_file="checkpoints/version_0/hparams.yaml",
)

model.model.save_pretrained("transformers/")
model.tokenizer.save_pretrained("transformers/")
