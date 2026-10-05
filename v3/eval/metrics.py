"""Corpus-level MT quality metrics, all computed on detokenized plain text
(every tokenizer's decode() returns normal whitespace text) so scores stay
comparable across word/BPE/WordPiece/char variants -- computing on raw
subword/character sequences would silently produce incomparable numbers.

BLEU: n-gram precision, standard MT metric, sensitive to exact matches.
chrF++: character n-gram F-score, more robust to morphological variation
  than BLEU -- useful here since French inflects more than English and a
  correct-but-differently-inflected word tanks BLEU harder than chrF++.
TER: edit distance (insert/delete/substitute/shift) to turn the hypothesis
  into the reference, normalized by reference length -- lower is better
  (unlike BLEU/chrF++), included specifically because that inversion is a
  real thing to get right when building a report, not just another score.
"""
import sacrebleu


def corpus_bleu(hypotheses: list, references: list) -> float:
    return sacrebleu.corpus_bleu(hypotheses, [references]).score


def corpus_chrf(hypotheses: list, references: list) -> float:
    return sacrebleu.corpus_chrf(hypotheses, [references], word_order=2).score  # chrF++


def corpus_ter(hypotheses: list, references: list) -> float:
    return sacrebleu.corpus_ter(hypotheses, [references]).score


def all_metrics(hypotheses: list, references: list) -> dict:
    return {
        "bleu": corpus_bleu(hypotheses, references),
        "chrf": corpus_chrf(hypotheses, references),
        "ter": corpus_ter(hypotheses, references),
    }
