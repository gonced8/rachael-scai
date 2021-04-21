"""
Create dataset files (train and test) (json).

Functions:

"""

# Standard Modules
import json
import os
import re

# Other Modules
import numpy as np
import pandas as pd

# Custom Modules

# Global Variables
context_token = "<context>"
# history_token = "<history>"
question_token = "<question>"


def process_data(data):
    dataset = []

    for sample in data["data"]:
        context = sample["story"]
        questions = sample["questions"]
        answers = sample["answers"]

        # history = ""

        for question, answer in zip(questions, answers):
            if question["turn_id"] != answer["turn_id"]:
                print(
                    "question and answer turn ids don't match for sample", sample["id"]
                )

            question = question["input_text"]
            answer = answer["input_text"]

            dataset.append(
                {
                    "src": context_token + context
                    # + history_token + history
                    + question_token + question,
                    "tgt": answer,
                }
            )

            # history += question

    return dataset


def get_dataset_name(dataset_folder, filename):
    if "train" in filename:
        filename = "train.json"
    elif "test" in filename or "dev" in filename or "val" in filename:
        filename = "test.json"
    return os.path.join(dataset_folder, filename)


def save_dataset(dataset, path):
    # Save as JSON
    with open(path, "w") as f:
        json.dump(dataset, f, indent=4, separators=None)
    return


if __name__ == "__main__":
    data_folder = "data"
    dataset_folder = "dataset"
    file_format = ".json"

    for (dirpath, _, filenames) in os.walk(data_folder):
        for filename in filenames:
            if ".json" in filename:
                # Read data
                data = pd.read_json(
                    os.path.join(dirpath, filename),
                )

                dataset = process_data(data)

                path = get_dataset_name(dataset_folder, filename)

                # Save dataset
                save_dataset(dataset, path)
