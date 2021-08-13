from argparse import ArgumentParser
import yaml

from model import get_model, get_data
from trainer import build_trainer


def main(args, hparams):
    if args.mode == "train":
        if args.from_checkpoint:
            print(f"Training from checkpoint: {args.from_checkpoint}")
            model = get_model(hparams.model_name).load_from_checkpoint(
                args.from_checkpoint
            )
        else:
            print("No checkpoint provided. Training from scratch.")
            model = get_model(hparams.model_name)(hparams)

        data = get_data(hparams.data_name)(model.hparams, model.tokenizer)

        trainer = build_trainer(model.hparams)
        trainer.fit(model, data)

    elif args.mode == "inference":
        if not args.from_checkpoint:
            print("No checkpoint provided. Please provide one.")
            return
        print(f"Inference mode from checkpoint: {args.from_checkpoint}")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--config", type=str, help="Configuration file for training the model."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "inference"],
        help="Mode for running this script.",
    )
    parser.add_argument(
        "--from_checkpoint",
        type=str,
        default=None,
        help="Path to load a model from checkpoint.",
    )
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            hparams = yaml.full_load(f)

    main(args, hparams)
