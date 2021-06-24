import csv
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
    username = data["From"].iloc[0]
    sender = ["USER:" if sender == username else "AGENT:" for sender in data["From"]]
    text = data.iloc[:, -1].to_list()
    text = "\n".join([" ".join(line) for line in zip(sender, text)])
    return text


def write_json(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    data_folder = "/tmp/gecr/ubuntu-ranking-dataset-creator/src/dialogs"
    output_folder = "/tmp/gecr/pegasus-qa/data/ubuntu/documents"

    i = 0

    for doc_id, data in read_files(data_folder):
        text = format_text(data)
        doc = {"id": doc_id, "contents": text}
        output_filename = os.path.join(output_folder, doc_id + ".json")
        write_json(output_filename, doc)

        if i % 1000 == 0:
            print(i)
        i += 1
