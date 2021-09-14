import json
import os

from pyserini.search import SimpleSearcher
from pyserini.dsearch import SimpleDenseSearcher, TctColBertQueryEncoder
from pyserini.hsearch import HybridSearcher
import torch
import tqdm
from transformers import T5ForConditionalGeneration, T5TokenizerFast


def test(model, conf):
    data_path = os.path.join(conf["test_dataset"])
    with open(data_path, "r") as f:
        data = json.load(f)

    # Setup question rewriting model
    tokenizer_t5 = T5TokenizerFast.from_pretrained(conf["rewrite_model_name"])
    t5 = T5ForConditionalGeneration.from_pretrained(conf["rewrite_model_name"]).cuda()
    t5.eval()

    # Setup retrieval model
    if "passages" in conf:
        ssearcher = SimpleSearcher(conf["passages"])
        ssearcher.set_bm25(0.82, 0.68)
    else:
        ssearcher = None

    if "dense_passages" in conf:
        encoder = TctColBertQueryEncoder(
            # "castorini/tct_colbert-msmarco", device="cuda"
            "sentence-transformers/msmarco-distilbert-base-v3",
            device="cuda",
        )
        dsearcher = SimpleDenseSearcher(
            # "data/qrecc/passages-dense-index-anserini", encoder
            "data/qrecc/passages-better-dense-index-anserini",
            encoder,
        )
    else:
        dsearcher = None

    if ssearcher is not None and dsearcher is not None:
        searcher = HybridSearcher(dsearcher, ssearcher)
    elif ssearcher is not None:
        searcher = ssearcher
    else:
        searcher = dsearcher

    # Setup answer generation model
    tokenizer_pegasus = model.tokenizer
    pegasus = model.model.cuda().eval()
    pegasus.eval()

    torch.no_grad()

    # TEST
    results = []

    conversation = None
    for sample in tqdm.tqdm(data, desc="Testing..."):
        if conversation != sample["Conversation_no"]:
            conversation = sample["Conversation_no"]
            history = []

        history.append(sample["Question"])

        # REWRITE
        while True:
            # Tokenize
            rewrite_input = " ||| ".join(history[-conf["rewrite_max_history"] :])
            # print(rewrite_input)

            rewrite_input_ids = tokenizer_t5.encode(
                rewrite_input,
                truncation=False,
                # truncation=True,
                # max_length=conf["rewrite_max_input_length"],
                return_tensors="pt",
            )

            if len(rewrite_input_ids) <= conf["rewrite_max_input_length"]:
                break
            else:
                del history[: -conf["rewrite_max_history"] + 1]

        rewrite_input_ids = rewrite_input_ids.cuda()

        output = t5.generate(
            rewrite_input_ids,
            max_length=conf["rewrite_max_output_length"],
            do_sample=True,
        )

        model_rewrite = tokenizer_t5.batch_decode(
            output,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

        history[-1] = model_rewrite

        # RETRIEVE
        candidates = searcher.search(model_rewrite)
        model_passages = {
            hit.docid: hit.score for hit in candidates[: conf["max_candidates"]]
        }

        # GENERATE
        docs = [searcher.doc(docid) for docid in model_passages]
        passages = [json.loads(doc.raw())["contents"] for doc in docs]

        context = "\n".join(history[-conf["max_history"] :])

        generate_input = "\n\n".join([context] + passages)
        # print(generate_input)
        generate_input = generate_input.replace("\n", "<n>")

        generate_input_ids = tokenizer_pegasus.encode(
            generate_input,
            truncation=True,
            max_length=conf["max_input_length"],
            return_tensors="pt",
        ).cuda()

        output = pegasus.generate(
            generate_input_ids,
            max_length=conf["max_output_length"],
            do_sample=True,
            no_repeat_ngram_size=conf["no_repeat_ngram_size"],
        )

        model_answer = tokenizer_pegasus.batch_decode(
            output,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]
        # print(model_answer)
        # input()

        history.append(model_answer)

        results.append(
            {
                "Conversation_no": sample["Conversation_no"],
                "Turn_no": sample["Turn_no"],
                "Turn_no": sample["Question"],
                "Model_rewrite": model_rewrite,
                "Model_passages": model_passages,
                "Model_answer": model_answer,
            }
        )

        print(results[-1])

    filename = "run.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
        print(f"Saved test output to: {filename}")
