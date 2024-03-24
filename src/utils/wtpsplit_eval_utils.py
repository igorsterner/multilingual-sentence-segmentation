import subprocess
import unicodedata

import numpy as np
import re
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)

from utils.wtpsplit_utils import (
    Constants,
    lang_code_to_lang,
    reconstruct_sentences,
)


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

ERSATZ_LANGUAGES = {
    "ar",
    "cs",
    "de",
    "en",
    "es",
    "et",
    "fi",
    "fr",
    "gu",
    "hi",
    "iu",
    "ja",
    "kk",
    "km",
    "lt",
    "lv",
    "pl",
    "ps",
    "ro",
    "ru",
    "ta",
    "tr",
    "zh",
}


class LanguageError(ValueError):
    pass


def ersatz_sentencize(
    lang_code,
    text,
    infile="notebooks/data/tmp/in.tmp",
    outfile="notebooks/data/tmp/out.tmp",
):
    if lang_code not in ERSATZ_LANGUAGES:
        raise LanguageError(f"ersatz does not support {lang_code}")
    open(infile, "w").write(text)

    subprocess.check_output(
        f"cat {infile} | ersatz --quiet -m default-multilingual > {outfile}",
        shell=True,
    )

    return reconstruct_sentences(text, open(outfile).readlines())


def pysbd_sentencize(lang_code, text):
    import pysbd

    try:
        return reconstruct_sentences(
            text, pysbd.Segmenter(language=lang_code, clean=False).segment(text)
        )
    except ValueError:
        raise LanguageError(f"pysbd does not support {lang_code}")


SPACY_LANG_TO_DP_MODEL = {
    "ca": "ca_core_news_sm",
    "zh": "zh_core_web_sm",
    "hr": "hr_core_news_sm",
    "da": "da_core_news_sm",
    "nl": "nl_core_news_sm",
    "en": "en_core_web_sm",
    "fi": "fi_core_news_sm",
    "fr": "fr_core_news_sm",
    "de": "de_core_news_sm",
    "el": "el_core_news_sm",
    "it": "it_core_news_sm",
    "ja": "ja_core_news_sm",
    "ko": "ko_core_news_sm",
    "lt": "lt_core_news_sm",
    "mk": "mk_core_news_sm",
    "nb": "nb_core_news_sm",
    "pl": "pl_core_news_sm",
    "pt": "pt_core_news_sm",
    "ro": "ro_core_news_sm",
    "ru": "ru_core_news_sm",
    "es": "es_core_news_sm",
    "sv": "sv_core_news_sm",
    "uk": "uk_core_news_sm",
}


def spacy_sent_sentencize(lang_code, text):
    import spacy

    try:
        nlp = spacy.blank(lang_code)
        nlp.add_pipe("sentencizer")

        if lang_code == "ja":
            # spacy uses SudachiPy for japanese, which has a length limit:
            # https://github.com/WorksApplications/sudachi.rs/blob/c7d20b22c68bb3f6585351847ae91bc9c7a61ec5/sudachi/src/input_text/buffer/mod.rs#L124-L126
            # so we need to chunk the input and sentencize the chunks separately
            chunksize = 10_000
            chunks = []
            for i in range(0, len(text), chunksize):
                chunks.append(text[i : i + chunksize])

            assert sum(len(c) for c in chunks) == len(text)
            return reconstruct_sentences(
                text, [str(s) for c in chunks for s in nlp(c).sents]
            )

        return reconstruct_sentences(text, list([str(s) for s in nlp(text).sents]))
    except ImportError:
        raise LanguageError(f"spacy_sent does not support {lang_code}")


def spacy_dp_sentencize(lang_code, text):
    import spacy

    try:
        nlp = spacy.load(SPACY_LANG_TO_DP_MODEL[lang_code], disable=["ner"])

        if lang_code == "ja":
            # spacy uses SudachiPy for japanese, which has a length limit:
            # https://github.com/WorksApplications/sudachi.rs/blob/c7d20b22c68bb3f6585351847ae91bc9c7a61ec5/sudachi/src/input_text/buffer/mod.rs#L124-L126
            # so we need to chunk the input and sentencize the chunks separately
            chunksize = 10_000
            chunks = []
            for i in range(0, len(text), chunksize):
                chunks.append(text[i : i + chunksize])

            assert sum(len(c) for c in chunks) == len(text)
            return reconstruct_sentences(
                text, [str(s) for c in chunks for s in nlp(c).sents]
            )

        return reconstruct_sentences(text, list([str(s) for s in nlp(text).sents]))
    except KeyError:
        raise LanguageError(f"spacy_dp does not support {lang_code}")


def punkt_sentencize(lang_code, text):
    from nltk.tokenize import sent_tokenize

    try:
        return reconstruct_sentences(
            text, sent_tokenize(text, language=lang_code_to_lang(lang_code).lower())
        )
    except LookupError:
        raise LanguageError(f"punkt does not support {lang_code}")
