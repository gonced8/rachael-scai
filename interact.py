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
    with open("config/example.yaml", "r") as f:
        hparams = yaml.full_load(f)
    hparams["model_name"] = "gonced8/pegasus-qa"
    model = Pegasus(hparams)

    hparams = model.hparams
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = model.tokenizer
    model = model.model.to(device)

    ssearcher = SimpleSearcher.from_prebuilt_index("msmarco-passage")
    encoder = TctColBertQueryEncoder("castorini/tct_colbert-msmarco")
    dsearcher = SimpleDenseSearcher.from_prebuilt_index(
        "msmarco-passage-tct_colbert-hnsw", encoder
    )
    hsearcher = HybridSearcher(dsearcher, ssearcher)

    my_searcher = SimpleSearcher("data/ubuntu/index/sparse")

    return hparams, device, tokenizer, model, ssearcher, hsearcher, my_searcher


hparams, device, tokenizer, model, ssearcher, hsearcher, my_searcher = init()

st.subheader("Question")
question = st.text_area("", "How do I restart my phone?")
# question_placeholder = st.empty()

st.subheader("Context")
retrieval = st.selectbox("Retrieval", ["None", "MSMARCO", "Ubuntu"], index=0)
candidates = st.slider("# of retrieved candidates", min_value=0, max_value=20, value=0)
context = st.text_area("Manual")
# context_placeholder = st.empty()

if st.button("Compute"):
    if retrieval != "None" and candidates > 0:
        if retrieval == "MSMARCO":
            hits = hsearcher.search(question)[:candidates]
            docs = [ssearcher.doc(hit.docid) for hit in hits]
            passages = [json.loads(doc.raw())["contents"] for doc in docs]
        elif retrieval == "Ubuntu":
            hits = my_searcher.search(question)[:candidates]
            passages = [json.loads(doc.raw)["contents"] for doc in hits]

        passages = [
            passage.encode("ascii", "ignore").decode("utf8") for passage in passages
        ]
        if context:
            passages.insert(0, context)
        context = "\n".join(passages)
        context = context.encode("ascii", "ignore").decode("utf8")

    src = (question + "\n" + context).replace("\n", "<n>")
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
    tgt_text = tokenizer.batch_decode(translated)[0].replace("<n>", "  \n")
    st.subheader("Answer")
    st.write(tgt_text)

    st.subheader("Model Input")
    st.write(input_text.replace("<n>", "  \n"))
