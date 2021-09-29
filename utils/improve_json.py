import json
import os
import sys

from pyserini.search import SimpleSearcher
from tqdm import tqdm
from unidecode import unidecode


def main(filenames):
    # Setup retrieval model
    searcher = SimpleSearcher("data/qrecc/passages-index-anserini")

    # Read input files
    test_filename = "data/qrecc/scai-qrecc21-test-questions.json"
    test_rewritten_filename = "data/qrecc/scai-qrecc21-test-questions-rewritten.json"

    with open(test_filename, "r") as f:
        test_data = json.load(f)

    with open(test_rewritten_filename, "r") as f:
        test_rewritten_data = json.load(f)

    # Loop through output files
    for filename in tqdm(filenames, desc="Processing results files"):
        # Read output file
        with open(filename, "r") as f:
            data = json.load(f)

        if len(data) != len(test_data):
            print(
                f"Error: mismatch between number of samples of {filename} ({len(data)}) and {test_filename} ({len(test_data)}). Ignoring this file..."
            )
            continue

        for test_input, test_rewritten_input, output in tqdm(
            zip(test_data, test_rewritten_data, data),
            total=len(data),
            desc="Processing samples",
        ):
            if (
                test_input["Conversation_no"] != output["Conversation_no"]
                or test_input["Turn_no"] != output["Turn_no"]
            ):
                print(
                    f"Error: mismatch of samples order of {filename} and {test_filename}. Ignoring this sample..."
                )
                continue

            if "Question" not in output:
                output["Question"] = test_input["Question"]

            if "Truth_rewrite" not in output:
                output["Truth_rewrite"] = test_rewritten_input["Question"]

            if "Passages_text" not in output:
                model_passages = output["Model_passages"]
                docs = [searcher.doc(docid) for docid in model_passages]
                output["Passages_text"] = {
                    doc.docid(): unidecode(json.loads(doc.raw())["contents"])
                    for doc in docs
                }

        # Get filename of new JSON
        new_filename = list(os.path.splitext(filename))
        new_filename[0] += "_new"
        new_filename = "".join(new_filename)

        # Save JSON
        with open(new_filename, "w") as f:
            json.dump(data, f, indent=2)
            print(f"Saved from {filename} to {new_filename}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1:])
    else:
        print("Please specify files to edit.")
        print(f"Example: {sys.argv[0]} file1.json file2.json")
