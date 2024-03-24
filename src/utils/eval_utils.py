import re
import unicodedata

import numpy as np
import spacy
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import pipeline
from wtpsplit import WtP

from utils.utils import Constants, reconstruct_sentences

wtp = WtP("wtp-canine-s-12l-no-adapters")
wtp.half().to("cuda")

pipe = pipeline(
    "token-classification",
    model="igorsterner/xlmr-multilingual-sentence-segmentation",
    stride=5,
    device=0,
)

spacy_nlp = spacy.load("xx_sent_ud_sm")


def preprocess_sentence(sentence):
    # right-to-left-mark
    sentence = sentence.replace(chr(8207), "")

    return re.sub(
        r"\s+", " ", unicodedata.normalize("NFKC", sentence.lstrip("-").strip())
    )


def get_labels(lang_code, sentences, after_space=True):
    separator = Constants.SEPARATORS[lang_code]
    text = separator.join(sentences)

    true_end_indices = np.cumsum(np.array([len(s) for s in sentences])) + np.arange(
        1, len(sentences) + 1
    ) * len(separator)
    # no space after last
    true_end_indices[-1] -= len(separator)

    if not after_space:
        true_end_indices -= len(separator) + 1

    labels = np.zeros(len(text) + 1)
    labels[true_end_indices] = 1

    return labels


def evaluate_sentences(lang_code, sentences, predicted_sentences):

    separator = Constants.SEPARATORS[lang_code]

    text = separator.join(sentences)

    assert len(text) == len("".join(predicted_sentences))

    labels = get_labels(lang_code, sentences)

    predicted_end_indices = np.cumsum(np.array([len(s) for s in predicted_sentences]))
    predictions = np.zeros_like(labels)
    predictions[predicted_end_indices] = 1

    return f1_score(labels, predictions), {
        "recall": recall_score(labels, predictions),
        "precision": precision_score(labels, predictions),
    }


class LanguageError(ValueError):
    pass


def spacy_multilingual_sentencize(text):

    return reconstruct_sentences(text, list([str(s) for s in spacy_nlp(text).sents]))


def xlmr_sentencize(text):

    output = pipe(text)

    pred_sentences = []
    start = 0
    for token in output:
        sentence = text[start : token["end"]]
        pred_sentences.append(sentence)
        start = token["end"]

    if start < len(text):
        pred_sentences.append(text[start:])

    return reconstruct_sentences(text, [str(s) for s in pred_sentences])


def wtpsplit_sententize(text):

    pred_sentences = wtp.split(text)

    return reconstruct_sentences(text, pred_sentences)
