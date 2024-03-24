import json
import pickle
from dataclasses import dataclass
from typing import List

from tqdm import tqdm
from transformers import HfArgumentParser

from utils.eval_utils import (
    LanguageError,
    evaluate_sentences,
    preprocess_sentence,
    spacy_multilingual_sentencize,
    wtpsplit_sententize,
    xlmr_sentencize,
)
from utils.utils import Constants


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

            sentences = [s for s in sentences if s]
            sentences = [preprocess_sentence(s) for s in sentences]
            text = Constants.SEPARATORS[lang_code].join(sentences)

            results[lang_code][dataset_name] = {}

            for f, name in [
                (xlmr_sentencize, "xlmr"),
                (wtpsplit_sententize, "wtpsplit"),
                (spacy_multilingual_sentencize, "spacy_multilingual"),
            ]:

                print(f"Running {name} on {dataset_name} in {lang_code}...")
                try:
                    results[lang_code][dataset_name][name] = evaluate_sentences(
                        lang_code, sentences, f(text)
                    )
                except LanguageError:
                    results[lang_code][dataset_name][name] = None

    with open(args.results_path, "w") as f:
        json.dump(results, f, indent=4)
