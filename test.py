import json
import os

from pyserini.search import SimpleSearcher
import torch
import tqdm

DEBUG = False


def test(model, conf):
    data_path = os.path.join(conf["test_dataset"])
    with open(data_path, "r") as f:
        data = json.load(f)

    rewrite = "rewritten" not in conf["test_dataset"] and conf.get(
        "rewrite_model_name", []
    )

    if rewrite:
        from transformers import T5ForConditionalGeneration, T5TokenizerFast

        # Setup question rewriting model
        tokenizer_t5 = T5TokenizerFast.from_pretrained(conf["rewrite_model_name"])
        t5 = T5ForConditionalGeneration.from_pretrained(
            conf["rewrite_model_name"]
        ).cuda()
        t5.eval()

    # Setup retrieval model
    ssearcher = SimpleSearcher(conf["passages"])
    ssearcher.set_bm25(0.82, 0.68)

    if conf.get("dense_passages", []):
        from pyserini.dsearch import SimpleDenseSearcher, TctColBertQueryEncoder
        from pyserini.hsearch import HybridSearcher

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
        searcher = HybridSearcher(dsearcher, ssearcher)
    else:
        searcher = ssearcher

    # Setup re-ranker
    if conf.get("reranker", []):
        from pygaggle.rerank.base import Query, Text
        from pygaggle.rerank.transformer import DuoT5
        from pygaggle.rerank.base import hits_to_texts

        reranker = DuoT5()
    else:
        reranker = None

    # Setup answer generation model
    tokenizer_pegasus = model.tokenizer
    pegasus = model.model.cuda().eval()
    pegasus.eval()

    torch.no_grad()

    # TEST
    results = []

    conversation = None
    for sample in tqdm.tqdm(data, desc="Testing..."):
        result = {
            "Conversation_no": sample["Conversation_no"],
            "Turn_no": sample["Turn_no"],
            "Question": sample["Question"],
        }

        if conversation != sample["Conversation_no"]:
            conversation = sample["Conversation_no"]
            history = []
            original_history = []

        history.append(sample["Question"])
        original_history.append(sample["Question"])

        # REWRITE
        if rewrite:
            if DEBUG:
                print("rewrite")
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
            result["Model_rewrite"] = model_rewrite

        # RETRIEVE
        if DEBUG:
            print("retrieve")
        query = "\n".join(history[-conf["max_history"] :])
        hits = searcher.search(query, k=conf["max_candidates"])[
            : conf["max_candidates"]
        ]

        # RE-RANK
        if reranker is None:
            model_passages = {hit.docid: hit.score for hit in hits}
        else:
            if DEBUG:
                print("re-rank")
            query = Query(query)
            texts = hits_to_texts(hits)
            reranked = reranker.rerank(query, texts)[: conf["max_rerank_candidates"]]
            reranked.sort(reverse=True, key=lambda hit: hit.score)

            model_passages = {hit.metadata["docid"]: hit.score for hit in reranked}

        passages = [
            json.loads(ssearcher.doc(docid).raw())["contents"]
            for docid in model_passages
        ]
        result["Model_passages"] = model_passages

        # GENERATE
        if DEBUG:
            print("generate")

        if "original_history" in conf and conf["original_history"]:
            context = "\n".join(original_history[-conf["original_max_history"] :])
        else:
            context = query

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

        if DEBUG:
            model_input = tokenizer_pegasus.batch_decode(
                generate_input_ids,
            )[0]
            print(
                f"Length: {len(generate_input_ids[0])}\t Passages: {model_input.count('<n><n>')}"
            )

        model_answer = tokenizer_pegasus.batch_decode(
            output,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]
        # print(model_answer)
        # input()

        history.append(model_answer)
        original_history.append(model_answer)
        result["Model_answer"] = model_answer

        results.append(result)
        if DEBUG:
            print(result)

    filename = "run.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
        print(f"Saved test output to: {filename}")
