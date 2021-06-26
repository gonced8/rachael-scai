import csv
import itertools
import json
import os

import pandas as pd


def read_files(data_folder):
    for (dirpath, _, filenames) in os.walk(data_folder):
        for filename in filenames:
            if ".tsv" in filename:
                doc_id = (
                    os.path.split(dirpath)[-1] + "_" + os.path.splitext(filename)[0]
                )

                filepath = os.path.join(dirpath, filename)
                data = pd.read_csv(
                    filepath,
                    sep="\t",
                    names=["Time", "From", "To", "Message"],
                    dtype=str,
                    quoting=csv.QUOTE_NONE,
                    encoding="utf-8",
                    keep_default_na=False,
                )
                yield doc_id, data


def format_text(data):
    count = [sum(1 for _ in group) for _, group in itertools.groupby(data["From"])]
    iterator = iter(data["Message"])
    text = [[next(iterator) for _ in range(length)] for length in count]

    # If the first message is from the agent, delete it
    if data["To"].iloc[0]:
        del text[0]

    # If there is an ending message from the user, delete it
    if len(text) % 2 != 0:
        del text[-1]

    senders = itertools.cycle(["USER: ", "AGENT: "])
    text = "\n".join(
        sender + "\n".join(messages) for sender, messages in zip(senders, text)
    )

    return text


def write_json(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    data_folder = "../ubuntu-ranking-dataset-creator/src/dialogs"
    output_folder = "data/ubuntu/documents"

    i = 0

    for doc_id, data in read_files(data_folder):
        text = format_text(data)
        doc = {"id": doc_id, "contents": text}
        output_filename = os.path.join(output_folder, doc_id + ".json")
        write_json(output_filename, doc)

        if i % 1000 == 0:
            print(i)
        i += 1
