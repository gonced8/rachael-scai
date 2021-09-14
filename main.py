from argparse import ArgumentParser
import yaml

from model import get_model, get_data
from test import test
from trainer import build_trainer


def main(conf):
    if conf["from_checkpoint"]:
        print(f"Loading from checkpoint: {conf['from_checkpoint']}")
        model = get_model(conf["model_name"]).load_from_checkpoint(
            conf["from_checkpoint"]
        )
    else:
        print("No checkpoint provided.")
        model = get_model(conf["model_name"])(conf)

    if conf["mode"] == "train":
        data = get_data(conf["data_name"])(conf, model.tokenizer)
        trainer = build_trainer(model.hparams)
        trainer.fit(model, data)

    elif conf["mode"] == "test":
        test(model, conf)


if __name__ == "__main__":
    parser = ArgumentParser(description="Train and test model.")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Configuration file for training the model.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test"],
        required=True,
        help="Mode for running this script.",
    )
    parser.add_argument(
        "--from_checkpoint",
        type=str,
        help="Path to load a model from checkpoint.",
    )
    # parser.add_argument(
    #    "--input_dir",
    #    type=str,
    #    default="",
    #    help="Path to input directory that contains the questions.json input file and passages-index-anserini directory.",
    # )
    # parser.add_argument(
    #    "--output_dir",
    #    type=str,
    #    default="",
    #    help="Path to output directory where run.json will be written.",
    # )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        conf = yaml.safe_load(f)

    # Merge arguments and config
    conf.update(vars(args))

    main(conf)
