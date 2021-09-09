import json
import os

from pyserini.search import SimpleSearcher
import torch
import tqdm
from transformers import T5ForConditionalGeneration, T5TokenizerFast

def test(model, conf):
    data_path = os.path.join(conf["input_dir"], conf["test_dataset"])
    with open(data_path, "r") as f:
        data = json.load(f)

    tokenizer_t5 = T5TokenizerFast.from_pretrained(conf["rewrite_model_name"])
    t5 = T5ForConditionalGeneration.from_pretrained(conf["rewrite_model_name"]).cuda()

    searcher = SimpleSearcher(
        os.path.join(conf["input_dir"], conf["passages"])
    )
    searcher.set_bm25(0.82, 0.68)

    tokenizer_pegasus = model.tokenizer
    pegasus = model.model.cuda().eval()

    t5.eval()
    pegasus.eval()
    torch.no_grad()
    
    results = []

    conversation = None
    for sample in tqdm.tqdm(data, desc="Testing..."):
        if conversation != sample["Conversation_no"]:
            conversation = sample["Conversation_no"]
            history = []

        history.append(sample["Question"])

        # REWRITE
        rewrite_input = " ||| ".join(history[-conf["rewrite_max_history"]:])

        #print(rewrite_input)
        # TODO: protect for when input is too large
        rewrite_input_ids = tokenizer_t5.encode(
                    rewrite_input,
                    truncation=True,
                    max_length=conf["rewrite_max_input_length"],
                    return_tensors="pt",
                ).cuda()

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
                hit.docid: hit.score for hit in candidates[:conf["max_candidates"]]
                }

        # GENERATE
        docs = [searcher.doc(docid) for docid in model_passages]
        passages = [json.loads(doc.raw())["contents"] for doc in docs]

        context = "\n".join(history[-conf["max_history"]:])

        generate_input = "\n\n".join([context] + passages)
        #print(generate_input)
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
        #print(model_answer)
        #input()

        history.append(model_answer)

        results.append({
                "Conversation_no": sample["Conversation_no"],
                "Turn_no": sample["Turn_no"],
                "Model_rewrite": model_rewrite,
                "Model_passages": model_passages,
                "Model_answer": model_answer,
            })


    filename = "run.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
        print(f"Saved test output to: {filename}")





        







        

