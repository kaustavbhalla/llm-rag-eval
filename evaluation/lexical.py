from rouge_score import rouge_scorer
from nltk.util import ngrams
from collections import Counter
import nltk


nltk.download("punkt",     quiet=False)
nltk.download("punkt_tab", quiet=False)
nltk.download("stopwords", quiet=False)


Rscorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

def computeRougeScore(candidate: str, reference: str) -> dict:
    scores = Rscorer.score(reference, candidate)
    return {
        "rouge1_precision": round(scores["rouge1"].precision, 4),
        "rouge1_recall":    round(scores["rouge1"].recall,    4),
        "rouge1_f1":        round(scores["rouge1"].fmeasure,  4),
        "rouge2_precision": round(scores["rouge2"].precision, 4),
        "rouge2_recall":    round(scores["rouge2"].recall,    4),
        "rouge2_f1":        round(scores["rouge2"].fmeasure,  4),
        "rougeL_precision": round(scores["rougeL"].precision, 4),
        "rougeL_recall":    round(scores["rougeL"].recall,    4),
        "rougeL_f1":        round(scores["rougeL"].fmeasure,  4),
    }


def _ngram_precision_recall_f1(
    candidate_tokens: list[str],
    reference_tokens:  list[str],
    n: int,
) -> dict[str, float]:
    cand_ngrams = Counter(ngrams(candidate_tokens, n))
    ref_ngrams  = Counter(ngrams(reference_tokens,  n))

    overlap    = sum((cand_ngrams & ref_ngrams).values())
    cand_total = sum(cand_ngrams.values())
    ref_total  = sum(ref_ngrams.values())

    precision = overlap / cand_total if cand_total > 0 else 0.0
    recall    = overlap / ref_total  if ref_total  > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def compute_ngram(candidate: str, reference: str) -> dict:
    cand_tokens = nltk.word_tokenize(candidate.lower())
    ref_tokens  = nltk.word_tokenize(reference.lower())

    unigram = _ngram_precision_recall_f1(cand_tokens, ref_tokens, 1)
    bigram  = _ngram_precision_recall_f1(cand_tokens, ref_tokens, 2)
    trigram = _ngram_precision_recall_f1(cand_tokens, ref_tokens, 3)

    return {
        "unigram_precision": unigram["precision"],
        "unigram_recall":    unigram["recall"],
        "unigram_f1":        unigram["f1"],
        "bigram_precision":  bigram["precision"],
        "bigram_recall":     bigram["recall"],
        "bigram_f1":         bigram["f1"],
        "trigram_precision": trigram["precision"],
        "trigram_recall":    trigram["recall"],
        "trigram_f1":        trigram["f1"],
    }

