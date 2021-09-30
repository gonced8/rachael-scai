import random
import json
import os
import sys
import yaml

import pandas as pd
from pyserini.search import SimpleSearcher
from tqdm import tqdm
from transformers import T5TokenizerFast, PegasusTokenizerFast

n_evaluators = 4
n_passages = 100

rewrite_model_name = "castorini/t5-base-canard"
model_name = "google/pegasus-large"
index = "../data/qrecc/passages-index-anserini"
rewrite_max_input_length = 512
max_input_length = 1024


def main(filenames, conf):
    # Setup rewriter
    tokenizer_t5 = T5TokenizerFast.from_pretrained(rewrite_model_name)

    # Setup retrieval model
    searcher = SimpleSearcher(index)

    # Setup generation
    tokenizer_pegasus = PegasusTokenizerFast.from_pretrained(model_name)

    # Loop through output files
    for filename in tqdm(filenames, desc="Processing results files"):
        run = os.path.split(os.path.splitext(filename)[0])[-1].split("_")[0]

        # Read output file
        with open(filename, "r") as f:
            data = json.load(f)

        # Augment samples with input of each model
        conversation = None
        for i, sample in enumerate(
            tqdm(data, desc="Augmenting samples with input of each model")
        ):
            sample = data[i]

            new_sample = {
                "Id": i,
                "Conversation_no": sample["Conversation_no"],
                "Turn_no": sample["Turn_no"],
                "Question": sample["Question"],
                "Truth_rewrite": sample["Truth_rewrite"],
            }

            # Construct conversation history
            if conversation != sample["Conversation_no"]:
                conversation = sample["Conversation_no"]
                history = []

            history.append(sample["Question"])

            # REWRITE
            if conf[run]["rewrite"]:
                while True:
                    # Tokenize
                    rewrite_max_history = conf[run]["rewrite"]["history"]
                    rewrite_input = " ||| ".join(history[-rewrite_max_history:])

                    rewrite_input_ids = tokenizer_t5.encode(
                        rewrite_input,
                        truncation=False,
                        return_tensors="pt",
                    )

                    if len(rewrite_input_ids) <= rewrite_max_input_length:
                        break
                    else:
                        del history[: -rewrite_max_history + 1]

                rewrite_input = tokenizer_t5.batch_decode(
                    rewrite_input_ids,
                )[0]

                new_sample["Rewrite_input"] = rewrite_input
                new_sample["Model_rewrite"] = sample["Model_rewrite"]
                history[-1] = sample["Model_rewrite"]

            # RETRIEVAL
            retrieval_max_history = conf[run]["retrieval"]["history"]
            query = "\n".join(history[-retrieval_max_history:])
            new_sample["Query"] = query

            model_passages = sample["Model_passages"]
            passages = list(sample["Passages_text"].values())

            # Transform passages dict to single columns
            for n, ((docid, score), text) in enumerate(
                zip(model_passages.items(), passages), start=1
            ):
                new_sample[f"Passage{n}_id"] = docid
                new_sample[f"Passage{n}_score"] = score
                new_sample[f"Passage{n}_text"] = text

            # GENERATE
            max_history = retrieval_max_history
            context = "\n".join(history[-max_history:])

            generate_input = "\n\n".join([context] + passages)
            generate_input = generate_input.replace("\n", "<n>")

            generate_input_ids = tokenizer_pegasus.encode(
                generate_input,
                truncation=True,
                max_length=max_input_length,
                return_tensors="pt",
            ).cuda()

            generate_input = tokenizer_pegasus.batch_decode(generate_input_ids,)[
                0
            ].replace("<n>", "\n")

            new_sample["Model_input"] = generate_input
            new_sample["Model_answer"] = sample["Model_answer"]

            # Update sample with new_sample (more info)
            data[i] = new_sample

        samples = random.sample(data, n_evaluators * n_passages)

        # Create Excel
        for evaluator in range(n_evaluators):
            split = samples[evaluator * n_passages : (evaluator + 1) * n_passages]

            df = pd.DataFrame(split)

            # Get filename of new file
            new_filename = list(os.path.splitext(filename))
            new_filename = f"{new_filename[0]}_{evaluator}.xlsx"

            # Save file
            df.to_excel(new_filename, engine="xlsxwriter")
            print(f"Saved from {filename} to {new_filename}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open("config.yaml", "r") as f:
            conf = yaml.safe_load(f)

        main(sys.argv[1:], conf)
    else:
        print("Please specify files to edit.")
        print(f"Example: {sys.argv[0]} file1.json file2.json")
