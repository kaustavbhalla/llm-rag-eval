import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

@st.cache_resource(show_spinner=False)
def loadNLIModel():
    tokenizere = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()

    return tokenizere, model

def workingNLI(premise: str, hypothesis: str) -> dict[str, float]:
    tokenizer, model = loadNLIModel()

    inputs = tokenizer(
        premise, hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    probabilities = F.softmax(logits, dim=-1).squeeze().tolist()
    idToLabel = model.config.id2label

    return {idToLabel[i]: probabilities[i] for i in range(3)}
