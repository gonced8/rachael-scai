import random
import json
import os
import sys
import yaml

import pandas as pd
from pyserini.search import SimpleSearcher
from tqdm import tqdm
from transformers import T5TokenizerFast, PegasusTokenizerFast

examples_ids = {
    "rewriting success  retrieval success   generation success": ["2061_3"],
    "rewriting success  retrieval success   generation fail": ["1525_6"],
    "rewriting success  retrieval fail   generation success": ["1706_2"],
    "rewriting success  retrieval fail   generation fail": ["430_2"],
    "rewriting fail  retrieval success   generation success": ["941_7"],
    "rewriting fail  retrieval success   generation fail": ["85_2"],
    "rewriting fail  retrieval fail   generation success": ["407_5"],
    "rewriting fail  retrieval fail   generation fail": ["1285_2"],
}

rewrite_model_name = "castorini/t5-base-canard"
model_name = "google/pegasus-large"
index = "../data/qrecc/passages-index-anserini"
rewrite_max_input_length = 512
max_input_length = 1024


def get_turn_id(turn):
    return "%d_%d" % (turn["Conversation_no"], turn["Turn_no"])


def main(filenames, conf):
    # Setup rewriter
    tokenizer_t5 = T5TokenizerFast.from_pretrained(rewrite_model_name)

    # Setup retrieval model
    searcher = SimpleSearcher(index)

    # Setup generation
    tokenizer_pegasus = PegasusTokenizerFast.from_pretrained(model_name)

    # Read test ground truth
    with open("../data/qrecc/scai-qrecc21-test-ground-truth.json", "r") as f:
        test_ground_truth = json.load(f)

    test_ground_truth = {get_turn_id(sample): sample for sample in test_ground_truth}

    # Loop through output files
    for filename in tqdm(filenames, desc="Processing results files"):
        run = os.path.split(os.path.splitext(filename)[0])[-1].split("_")[0]

        # Read output file
        with open(filename, "r") as f:
            data = json.load(f)

        # Augment samples with input of each model
        conversation = None
        data_dict = {}
        for i, sample in enumerate(
            tqdm(data, desc="Augmenting samples with input of each model")
        ):
            sample = data[i]
            test = test_ground_truth[get_turn_id(sample)]

            new_sample = {
                "Conversation_no": sample["Conversation_no"],
                "Turn_no": sample["Turn_no"],
                "Question": sample["Question"],
                "Truth_rewrite": test["Truth_rewrite"],
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
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )[0]

                new_sample["Model_rewrite"] = sample["Model_rewrite"]
                new_sample["Rewrite_input"] = rewrite_input

                if conf[run]["rewrite"]["question"].lower() == "model_rewrite":
                    history[-1] = sample["Model_rewrite"]

            # RETRIEVAL
            retrieval_max_history = conf[run]["retrieval"]["history"]
            query = "\n".join(history[-retrieval_max_history:])
            new_sample["Query"] = query

            passages = list(sample["Passages_text"].values())

            new_sample["Truth_passages"] = test["Truth_passages"]
            new_sample["Model_passages"] = sample["Model_passages"]

            # for (docid, score), text in zip(model_passages.items(), passages):

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
            )

            generate_input = tokenizer_pegasus.batch_decode(
                generate_input_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )[0].replace("<n>", "\n")

            new_sample["Truth_answer"] = test["Truth_answer"]
            new_sample["Model_answer"] = sample["Model_answer"]
            new_sample["Model_input"] = generate_input

            if conf[run]["rewrite"] and conf[run]["rewrite"]["model_answer"]:
                history.append(sample["Model_answer"])

            # Update sample with new_sample (more info)
            data[i] = new_sample
            data_dict[get_turn_id(new_sample)] = new_sample

        # Get examples
        examples = {}

        for case, turnids in examples_ids.items():
            examples[case] = [
                data_dict[turnid] for turnid in turnids if turnid in data_dict
            ]

        # Save examples
        with open("examples.json", "w") as f:
            json.dump(examples, f, indent=2)
            print(f"Saved from {filename} to examples.json")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open("config.yaml", "r") as f:
            conf = yaml.safe_load(f)

        main(sys.argv[1:], conf)
    else:
        print("Please specify files to edit.")
        print(f"Example: {sys.argv[0]} file1.json file2.json")
