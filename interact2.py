import json
import os
import yaml

from pyserini.search import SimpleSearcher
from pyserini.dsearch import SimpleDenseSearcher, TctColBertQueryEncoder
from pyserini.hsearch import HybridSearcher
import torch
import streamlit as st
from unidecode import unidecode

from model.model import Pegasus


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def init():
    filename = "checkpoints/version_10/checkpoints/best.ckpt"
    model = Pegasus.load_from_checkpoint(
        filename,
        hparams_file="checkpoints/version_10/hparams.yaml",
    )

    hparams = model.hparams
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = model.tokenizer
    model = model.model.to(device)

    ssearcher = SimpleSearcher("data/ubuntu/index/sparse")

    return hparams, device, tokenizer, model, ssearcher, []


hparams, device, tokenizer, model, ssearcher, history = init()

st.subheader("Question")
question = st.text_area("", "hello, my hdd is not recognized")
# question_placeholder = st.empty()

st.subheader("Context")
candidates = st.slider("# of retrieved candidates", min_value=0, max_value=20, value=5)
# context_placeholder = st.empty()

if st.button("Reset"):
    for i in range(len(history) - 1, -1, -1):
        del history[i]

if st.button("Compute"):
    question = "USER: " + question.strip()
    history.append(question)
    joined_history = "\n".join(history)

    if candidates > 0:
        hits = ssearcher.search(joined_history)[:candidates]
        passages = [json.loads(doc.raw)["contents"] for doc in hits]

        context = "\n\n".join(passages)
    else:
        context = ""

    src = joined_history + "\n\n" + context
    src = src.replace("\n", "<n>")
    src = unidecode(src)

    batch = tokenizer(
        src,
        truncation=True,
        max_length=hparams.max_input_length,
        return_tensors="pt",
    ).to(device)

    input_text = tokenizer.batch_decode(batch["input_ids"])[0]
    question_truncated, *context_truncated = input_text.split("<n>")
    context_truncated = "  \n".join(context_truncated)
    question_truncated = question_truncated.replace("<n>", "  \n")
    # context_placeholder.write(context_truncated)
    # question_placeholder.write(question_truncated)

    translated = model.generate(
        **batch, length_penalty=1.0, repetition_penalty=1.2, no_repeat_ngram_size=10
    )
    answer = tokenizer.batch_decode(translated)[0].replace("<n>", "  \n")

    history.append(answer)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Model Input")
    st.write(input_text.replace("<n>", "  \n"))
