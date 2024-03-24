import json
from dataclasses import dataclass
from typing import List

from tqdm import tqdm
from transformers import HfArgumentParser

from utils.wtpsplit_eval_utils import (
    LanguageError,
    ersatz_sentencize,
    evaluate_sentences,
    preprocess_sentence,
    punkt_sentencize,
    pysbd_sentencize,
    spacy_dp_sentencize,
    spacy_sent_sentencize,
)
from utils.wtpsplit_utils import Constants
import pickle

@dataclass
class Args:
    eval_data_path: str = "../data/all_eval_data.pkl"
    results_path: str = "../data/results.json"
    include_langs: List[str] = None


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()

    # eval_data = torch.load(args.eval_data_path)
    with open(args.eval_data_path, "rb") as f:
        eval_data = pickle.load(f)

    results = {}

    for lang_code, lang_data in tqdm(eval_data.items()):

        results[lang_code] = {}

        for dataset_name, sentences in lang_data.items():
            results[lang_code][dataset_name] = {}

            clean_sentences = [preprocess_sentence(s) for s in sentences]

            text = Constants.SEPARATORS[lang_code].join(clean_sentences)

            for f, name in [
                (punkt_sentencize, "punkt"),
                # (spacy_dp_sentencize, "spacy_dp"),
                # (spacy_sent_sentencize, "spacy_sent"),
                # (pysbd_sentencize, "pysbd"),
                # (ersatz_sentencize, "ersatz"),
            ]:
                print(f"Running {name} on {dataset_name} in {lang_code}...")
                try:
                    results[lang_code][dataset_name][name] = evaluate_sentences(
                        lang_code, sentences, f(lang_code, text)
                    )
                except LanguageError:
                    results[lang_code][dataset_name][name] = None

    with open(args.results_path, "w") as f:
        json.dump(results, f, indent=4)