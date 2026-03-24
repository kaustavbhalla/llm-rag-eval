"""
Three backends ensembled:
  NLI        (weight 0.6) — checks if samples entail each sentence
  BERTScore  (weight 0.3) — semantic similarity between sentence and samples
  NGram      (weight 0.1) — lexical consistency
"""

import numpy as np
import torch
import streamlit as st
import spacy
from selfcheckgpt.modeling_selfcheck import (
    SelfCheckNLI,
    SelfCheckBERTScore,
    SelfCheckNgram,
)


@st.cache_resource(show_spinner=False)
def _load_spacy():
    # run once: python -m spacy download en_core_web_sm
    return spacy.load("en_core_web_sm")


@st.cache_resource(show_spinner=False)
def _load_selfcheck_backends():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nli        = SelfCheckNLI(nli_model="cross-encoder/nli-deberta-v3-base", device=device)
    bertscore  = SelfCheckBERTScore(rescale_with_baseline=True)
    ngram      = SelfCheckNgram(n=1)
    return nli, bertscore, ngram


def _split_sentences(text: str) -> list[str]:
    nlp = _load_spacy()
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def evaluate_hallucination(main_answer: str, samples: list[str]) -> dict:
    if not main_answer or not samples:
        return {"hallucination_score": None, "sentence_scores": [], "sentences_flagged": 0, "total_sentences": 0}

    sentences = _split_sentences(main_answer)
    if not sentences:
        return {"hallucination_score": None, "sentence_scores": [], "sentences_flagged": 0, "total_sentences": 0}

    selfcheck_nli, selfcheck_bertscore, selfcheck_ngram = _load_selfcheck_backends()

    nli_scores       = selfcheck_nli.predict(sentences=sentences, sampled_passages=samples)
    bertscore_scores = selfcheck_bertscore.predict(sentences=sentences, sampled_passages=samples)
    ngram_scores     = selfcheck_ngram.predict(sentences=sentences, passage=main_answer, sampled_passages=samples)

    sentence_scores = []
    for i, sent in enumerate(sentences):
        nli_s   = float(nli_scores[i])
        bert_s  = float(bertscore_scores[i])
        ngram_s = float(ngram_scores[i])
        ensemble = round(0.6 * nli_s + 0.3 * bert_s + 0.1 * ngram_s, 4)

        sentence_scores.append({
            "sentence":        sent,
            "nli_score":       round(nli_s,   4),
            "bertscore_score": round(bert_s,  4),
            "ngram_score":     round(ngram_s, 4),
            "ensemble_score":  ensemble,
            "risk_level": (
                "High"   if ensemble > 0.5  else
                "Medium" if ensemble > 0.25 else
                "Low"
            ),
        })

    aggregate = round(float(np.mean([s["ensemble_score"] for s in sentence_scores])), 4)

    return {
        "hallucination_score": aggregate,
        "sentence_scores":     sentence_scores,
        "sentences_flagged":   sum(1 for s in sentence_scores if s["ensemble_score"] > 0.5),
        "total_sentences":     len(sentences),
    }
