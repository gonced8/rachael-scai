import json
import os
import yaml

from pyserini.search import SimpleSearcher
import torch
import streamlit as st

from model.model import Pegasus


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def init():
    filename = "checkpoints/version_0/checkpoints/best.ckpt"
    if os.path.isfile(filename):
        model = Pegasus.load_from_checkpoint(
            filename,
            hparams_file="checkpoints/version_0/hparams.yaml",
        )
    else:
        with open("config/example.yaml", "r") as f:
            hparams = yaml.full_load(f)
        hparams["model_name"] = "gonced8/pegasus-qa"
        model = Pegasus(hparams)

    hparams = model.hparams
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = model.tokenizer
    model = model.model.to(device)

    searcher = SimpleSearcher.from_prebuilt_index("msmarco-passage")

    return hparams, device, tokenizer, model, searcher


hparams, device, tokenizer, model, searcher = init()

st.subheader("Context")
msmarco = st.slider("MSMARCO retrieved candidates", min_value=0, max_value=20, value=0)
context = st.text_area("Manual")
# context_placeholder = st.empty()

st.subheader("Question")
question = st.text_area("", "How do I restart my phone?")
# question_placeholder = st.empty()

if st.button("Compute"):
    if msmarco > 0:
        hits = searcher.search(question)
        passages = [json.loads(passage.raw)["contents"] for passage in hits[:msmarco]]
        if context:
            passages.insert(0, context)
        context = "\n".join(passages)

    src = (question + "\n" + context).replace("\n", "<n>")
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

    translated = model.generate(**batch)
    tgt_text = tokenizer.batch_decode(translated)[0].replace("<n>", "  \n")
    st.subheader("Answer")
    st.write(tgt_text)

    st.subheader("Model Input")
    st.write(input_text.replace("<n>", "  \n"))
