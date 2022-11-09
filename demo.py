import json
import os
import re
import yaml

from pyserini.search import SimpleSearcher
import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast
import streamlit as st
from unidecode import unidecode

from model import get_model


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def init():
    # Configuration
    with open("config/demo.yaml", "r") as f:
        conf = yaml.safe_load(f)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup question rewriting model
    tokenizer_t5 = T5TokenizerFast.from_pretrained(conf["rewrite_model_name"])
    t5 = T5ForConditionalGeneration.from_pretrained(conf["rewrite_model_name"])
    t5.to(device)
    t5.eval()

    # Setup retrieval model
    searcher = SimpleSearcher(conf["passages"])
    searcher.set_bm25(0.82, 0.68)

    # Setup answer generation model
    model = get_model(conf["model_name"]).load_from_checkpoint(conf["from_checkpoint"])
    tokenizer_pegasus = model.tokenizer
    pegasus = model.model
    pegasus.to(device)
    pegasus.eval()

    torch.no_grad()

    return {
        "conf": conf,
        "device": device,
        "tokenizer_t5": tokenizer_t5,
        "t5": t5,
        "searcher": searcher,
        "tokenizer_pegasus": tokenizer_pegasus,
        "pegasus": pegasus,
    }


@st.cache(allow_output_mutation=True)
def Text():
    return ["Q: "]


def compute_answer(model, history):
    # REWRITE
    while True:
        # Tokenize
        rewrite_input = " ||| ".join(history[-model["conf"]["rewrite_max_history"] :])

        rewrite_input_ids = model["tokenizer_t5"].encode(
            rewrite_input,
            truncation=False,
            return_tensors="pt",
        )

        if len(rewrite_input_ids) <= model["conf"]["rewrite_max_input_length"]:
            break
        else:
            del history[: -model["conf"]["rewrite_max_history"] + 1]

    rewrite_input_ids = rewrite_input_ids.to(model["device"])

    output = model["t5"].generate(
        rewrite_input_ids,
        max_length=model["conf"]["rewrite_max_output_length"],
        do_sample=True,
    )

    model_rewrite = model["tokenizer_t5"].batch_decode(
        output,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]

    history[-1] = model_rewrite

    # RETRIEVE
    query = "\n".join(history[-model["conf"]["max_history"] :])
    hits = model["searcher"].search(query, k=model["conf"]["max_candidates"])[
        : model["conf"]["max_candidates"]
    ]

    model_passages = {hit.docid: hit.score for hit in hits}
    passages = [
        json.loads(model["searcher"].doc(docid).raw())["contents"]
        for docid in model_passages
    ]

    # GENERATE
    context = query
    generate_input = "\n\n".join([context] + passages)

    generate_input = generate_input.encode("ascii", "ignore").decode("utf8")
    generate_input = unidecode(generate_input)

    generate_input = generate_input.replace("\n", "<n>")

    generate_input_ids = (
        model["tokenizer_pegasus"]
        .encode(
            generate_input,
            truncation=True,
            max_length=model["conf"]["max_input_length"],
            return_tensors="pt",
        )
        .to(model["device"])
    )

    model_input = model["tokenizer_pegasus"].batch_decode(
        generate_input_ids,
    )[0]
    model_input = model_input.replace("<n>", "\n")
    text_passages = model_input[model_input.find("\n\n") + 1 :]

    output = model["pegasus"].generate(
        generate_input_ids,
        max_length=model["conf"]["max_output_length"],
        do_sample=True,
        no_repeat_ngram_size=model["conf"]["no_repeat_ngram_size"],
    )

    model_answer = model["tokenizer_pegasus"].batch_decode(
        output,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]

    history.append(model_answer)

    return model_rewrite, text_passages, model_answer


st.title('Ask Me "Anything"!')
st.header("Conversational Question Answering")

st.subheader("https://ama.goncaloraposo.com")

model = init()

left, right = st.columns(2)
compute = left.button("Compute")
reset = right.button("Reset")

text_cache = Text()
textbox = st.empty()

if reset:
    print("reset")
    text_cache[0] = "Q: "

if compute:
    print("compute")
    history = text_cache[0]
    history = history.split("\n")

    if any([sentence.startswith("R: ") for sentence in history]):
        marker = "R: "
    else:
        marker = "Q: "

    for i, sentence in enumerate(history):
        if sentence.startswith("Q") and "=>" in sentence:
            history[i] = "Q: " + sentence.split("=>")[-1].strip()

    history = [re.sub(r"^.*:\s?", "", sentence) for sentence in history]
    print(history)

    # If there is a new question
    if history[-1]:
        rewrite, passages, answer = compute_answer(model, history)
        text_cache[0] += f" => {rewrite}\nA: {answer}\nQ: "

        st.caption("Retrieved Passages")
        st.write(passages)

text_cache[0] = textbox.text_area("Conversation", text_cache[0], height=480)
