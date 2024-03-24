import glob
import os
import pickle
from dataclasses import dataclass

import conllu
import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import HfArgumentParser

from utils.denglisch_utils import Corpus
from utils.wtpsplit_eval_utils import preprocess_sentence
from utils.wtpsplit_utils import Constants

UD_TREEBANK_PATH = (
    "../data/ud-treebanks-v2.13"  # source: https://universaldependencies.org/#download
)
ERSATZ_DATA_PATH = "../data/ersatz-test-suite/segmented"  # source: https://github.com/rewicks/ersatz-test-suite

# copied from Table 8 in https://aclanthology.org/2021.acl-long.309.pdf
ERSATZ_TEST_DATASETS = {
    "ar": "iwsltt2017.ar",
    "cs": "wmt20.cs-en.cs",
    "de": "wmt20.de-en.de",
    "en": "wsj.03-06.en",
    "es": "wmt13.es-en.es",
    "et": "wmt18.et-en.et",
    "fi": "wmt19.fi-en.fi",
    "fr": "wmt20.fr-de.fr",
    "gu": "wmt19.gu-en.gu",
    "hi": "wmt14.hi-en.hi",
    "iu": "wmt20.iu-en.iu",
    "ja": "wmt20.ja-en.ja",
    "kk": "wmt19.kk-en.kk",
    "km": "wmt20.km-en.km",
    "lt": "wmt19.lt-en.lt",
    "lv": "wmt17.lv-en.lv",
    "pl": "wmt20.pl-en.pl",
    "ps": "wmt20.ps-en.ps",
    "ro": "wmt16.ro-en.ro",
    "ru": "wmt20.ru-en.ru",
    "ta": "wmt20.ta-en.ta",
    "tr": "wmt18.tr-en.tr",
    "zh": "wmt20.zh-en.zh",
}

DENGLISCH_PATH = "../data/denglisch/Manu_corpus.csv"
DENGLISCH_PATH = "/home/is473/rds/hpc-work/4X1/multilingual-sentence-segmentation/data/denglisch/Manu_corpus.csv"


@dataclass
class Args:
    output_file: str = "../data/all_eval_data.pkl"
    cache_dir: str = "../data/cache/"


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()

    eval_data = {lang_code: {} for lang_code in Constants.LANGINFO.index}

    # Ersatz data
    for lang_code in tqdm(Constants.LANGINFO.index):
        if lang_code in ERSATZ_TEST_DATASETS:
            ersatz_sentences = [
                preprocess_sentence(line)
                for line in open(
                    os.path.join(
                        ERSATZ_DATA_PATH, lang_code, ERSATZ_TEST_DATASETS[lang_code]
                    )
                )
            ]
            eval_data[lang_code]["ersatz"] = ersatz_sentences

    # UD + OPUS100 sentences
    for lang_code in tqdm(Constants.LANGINFO.index):
        opus_dset_name = Constants.LANGINFO.loc[lang_code, "opus100"]

        if opus_dset_name not in (np.nan, None):
            other_lang_code = set(opus_dset_name.split("-")) - {lang_code}
            assert len(other_lang_code) == 1
            other_lang_code = other_lang_code.pop()

            dset_args = ["opus100", opus_dset_name]

            try:
                opus100_sentences = [
                    preprocess_sentence(sample[lang_code])
                    for sample in load_dataset(
                        *dset_args, split="test", cache_dir=args.cache_dir
                    )["translation"]
                    if sample[lang_code].strip() != sample[other_lang_code].strip()
                ]
                eval_data[lang_code]["opus100"] = opus100_sentences
            except:
                opus100_sentences = [
                    preprocess_sentence(sample[lang_code])
                    for sample in load_dataset(
                        *dset_args, split="train", cache_dir=args.cache_dir
                    )["translation"]
                    if sample[lang_code].strip() != sample[other_lang_code].strip()
                ]
                eval_data[lang_code]["opus100"] = opus100_sentences

        if Constants.LANGINFO.loc[lang_code, "ud"] not in (np.nan, None):
            ud_data = conllu.parse(
                open(
                    glob.glob(
                        os.path.join(
                            UD_TREEBANK_PATH,
                            Constants.LANGINFO.loc[lang_code, "ud"],
                            "*-ud-test.conllu",
                        )
                    )[0]
                ).read()
            )

            ud_sentences = [
                preprocess_sentence(sentence.metadata["text"]) for sentence in ud_data
            ]
            eval_data[lang_code]["ud"] = ud_sentences

    # DengLisch

    denglisch_corpus = Corpus(DENGLISCH_PATH)
    all_tokens, all_labels = denglisch_corpus.get_sentences()

    denglisch_sentences = []

    for tokens, labels in tqdm(zip(all_tokens, all_labels)):
        sentence_tokens = []
        for token, label in zip(tokens, labels):
            if token and token != "$newline$":
                sentence_tokens.append(token)
            if label == "<EOS>" or label == "<EOP>":
                sentence_text = preprocess_sentence(" ".join(sentence_tokens))
                denglisch_sentences.append(sentence_text)
                sentence_tokens = []

    print(len(denglisch_sentences))

    eval_data["de"]["denglisch"] = denglisch_sentences

    with open(args.output_file, "wb") as f:
        pickle.dump(eval_data, f)
