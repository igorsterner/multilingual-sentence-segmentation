import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from cached_property import cached_property

# same as in CANINE
PRIMES = [31, 43, 59, 61, 73, 97, 103, 113, 137, 149, 157, 173, 181, 193, 211, 223]


class ConstantsClass:
    NEWLINE_INDEX = 0
    AUX_OFFSET = 1

    @cached_property
    def ROOT_DIR(self):
        return Path(os.path.abspath(os.path.join(os.path.dirname(__file__))))

    @cached_property
    def LANGINFO(self):
        return pd.read_csv(
            "../data/language_info.csv",
            index_col=0,
        )

    @cached_property
    def LANG_CODE_TO_INDEX(self):
        return {lang: i for i, lang in enumerate(Constants.LANGINFO.index)}

    @cached_property
    def SEPARATORS(self):
        return {
            lang: ("" if row["no_whitespace"] else " ")
            for lang, row in Constants.LANGINFO.iterrows()
        }


Constants = ConstantsClass()


@dataclass
class LabelArgs:
    newline_remove_prob: float = 1.0
    auxiliary_remove_prob: float = 0.5
    hyphen_remove_prob: float = 0.9
    newline_whitespace_prob: float = 0.99
    hyphen_smooth_prob: float = 0.9
    newline_chars: List[str] = field(default_factory=lambda: ["\n"])
    auxiliary_chars: List[str] = field(
        default_factory=lambda: Constants.PUNCTUATION_CHARS.copy()
    )
    hyphen_chars: List[str] = field(default_factory=lambda: ["-", "‐"])
    use_auxiliary: bool = False


def get_label_dict(label_args):
    label_dict = {}

    for i, c in enumerate(label_args.auxiliary_chars):
        label_dict[ord(c)] = 1 + Constants.AUX_OFFSET + i

    for c in label_args.newline_chars:
        label_dict[ord(c)] = 1 + Constants.NEWLINE_INDEX

    return label_dict


def sigmoid(x):
    return 1 / (1 + np.exp(-x.astype(np.float32)))  # fp32 for better precision


def encode(text):
    return [ord(c) for c in text]


def hash_encode(encoding, num_hashes=8, num_buckets=8192):
    if num_hashes > len(PRIMES):
        raise ValueError(f"`num_hashes` must be <= {len(PRIMES)}")

    hash_ids = np.zeros((len(encoding), num_hashes), dtype=np.int64)
    for i in range(num_hashes):
        shard_ids = (encoding + 1) * PRIMES[i]
        hash_ids[:, i] = shard_ids % num_buckets

    return hash_ids


def label(input_ids, label_dict):
    return [label_dict.get(input_id, 0) for input_id in input_ids[1:]] + [0]


def lang_code_to_lang(lang_code):
    from iso639 import languages

    if lang_code == "el":
        return "Greek"
    elif lang_code == "tl":
        return "Filipino"
    elif lang_code == "ug":
        return "Uyghur"

    try:
        return languages.get(alpha2=lang_code).name
    except KeyError:
        return languages.get(part3=lang_code).name


def indices_to_sentences(text, indices, strip_whitespace=False):
    sentences = []

    offset = 0
    idx = 0
    for idx in indices:
        idx = idx + 1
        while idx < len(text) and text[idx].isspace():
            idx += 1

        sentence = text[offset:idx]
        if strip_whitespace:
            # NB: I would have thought that this is slower than
            # adjusting the start and end indices since there are
            # two string copies, but it seems to be faster
            # (at least on short strings). more reason to port to Rust?
            sentence = sentence.strip()

        if len(sentence) > 0:
            sentences.append(sentence)

        offset = idx

    if idx != len(text):
        last_sentence = text[idx:]
        if strip_whitespace:
            last_sentence = last_sentence.strip()

        if len(last_sentence) > 0:
            sentences.append(last_sentence)

    return sentences


def reconstruct_sentences(text, partial_sentences):
    # for consistency
    partial_sentences = [x.strip() for x in partial_sentences]

    fixed_sentences = []
    i = 0
    for x in partial_sentences:
        try:
            idx = i + text[i:].index(x[:-1])
        except ValueError:
            reduced = x[:-2]

            while len(reduced) > 0 and not reduced.isspace():
                idx = i + text[i:].find(reduced)

                if idx == -1:
                    reduced = reduced[:-1]
                else:
                    break

            if idx == -1:
                raise ValueError("irrecoverable error in reconstruction")

        if idx > i:
            fixed_sentences.append(text[i:idx])
            i = idx

    if i != len(text):
        fixed_sentences.append(text[i:])

    return fixed_sentences
