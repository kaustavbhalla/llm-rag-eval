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

def NLIInference(premise: str, hypothesis: str) -> dict[str, float]:
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

def NLIInferenceChunked(premise: str, hypothesis: str, chunk_words: int = 350) -> dict[str, float]:
    words = premise.split()

    if len(words) <= chunk_words:
        return NLIInference(premise, hypothesis)
    
    chunks     = [" ".join(words[i:i + chunk_words]) for i in range(0, len(words), chunk_words)]
    all_scores = [NLIInference(chunk, hypothesis) for chunk in chunks]

    return {
        "entailment":    max(s["entailment"]    for s in all_scores),
        "neutral":       sum(s["neutral"]        for s in all_scores) / len(all_scores),
        "contradiction": max(s["contradiction"]  for s in all_scores),
    }


def semanticEval(
    llm_answer: str,
    correct_answer: str,
    retrieved_context: str,
) -> dict:
    acc_scores  = NLIInference(premise=correct_answer, hypothesis=llm_answer)

    comp_scores = NLIInference(premise=llm_answer,     hypothesis=correct_answer)

    grnd_scores = NLIInferenceChunked(premise=retrieved_context, hypothesis=llm_answer)

    return {
        "accuracy":            round(acc_scores["entailment"],           4),
        "completeness":        round(comp_scores["entailment"],          4),
        "truthfulness":        round(1.0 - acc_scores["contradiction"],  4),
        "groundedness":        round(grnd_scores["entailment"],          4),
        "contradiction_score": round(acc_scores["contradiction"],        4),
    }
