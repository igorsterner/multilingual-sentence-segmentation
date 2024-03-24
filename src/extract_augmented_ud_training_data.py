import glob
import os
from dataclasses import dataclass

import random
import pickle
import conllu
import numpy as np
from tqdm.auto import tqdm
from transformers import HfArgumentParser

from utils.wtpsplit_eval_utils import preprocess_sentence
from utils.wtpsplit_utils import Constants

UD_TREEBANK_PATH = "../data/ud-treebanks-v2.13"  # source: https://universaldependencies.org/#download

langs = Constants.LANGINFO.index


@dataclass
class Args:
    output_dir: str = "../data/preprocessed_training_data"


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()

    ud_data = {lang_code: {} for lang_code in langs}

    print("Starting to load UD data")

    for lang_code in tqdm(langs):

        if Constants.LANGINFO.loc[lang_code, "ud"] not in (np.nan, None):
            try:
                ud_dev_data = conllu.parse(
                    open(
                        glob.glob(
                            os.path.join(
                                UD_TREEBANK_PATH,
                                Constants.LANGINFO.loc[lang_code, "ud"],
                                "*-ud-dev.conllu",
                            )
                        )[0]
                    ).read()
                )
            except:
                ud_dev_sentences = []

            ud_dev_sentences = [
                preprocess_sentence(sentence.metadata["text"])
                for sentence in ud_dev_data
            ]

            try:
                ud_train_data = conllu.parse(
                    open(
                        glob.glob(
                            os.path.join(
                                UD_TREEBANK_PATH,
                                Constants.LANGINFO.loc[lang_code, "ud"],
                                "*-ud-train.conllu",
                            )
                        )[0]
                    ).read()
                )
            except:
                ud_train_sentences = []
                
            ud_train_sentences = [
                preprocess_sentence(sentence.metadata["text"])
                for sentence in ud_train_data
            ]

            p_casing = 1 / 3
            p_punct = 1 / 2
            p_add_full_stop = 1 / 100
            p_add_exclamation_mark = 1 / 5
            p_add_question_mark = 1 / 10
            p_add_semi_colon = 1 / 100

            corrupted_ud_dev_sentences = []
            for sentence in ud_dev_sentences:

                if len(sentence) < 2:
                    continue

                if random.uniform(0, 1) < p_casing:
                    sentence = sentence[0].lower() + sentence[1:]

                if random.uniform(0, 1) < p_punct:
                    sentence = sentence.rstrip(".?!;。！？")

                if len(sentence) < 2:
                    continue

                if (sentence[-1] == "." and sentence[-2] != ".") or (
                    sentence[-1] == "。" and sentence[-2] != "。"
                ):
                    while random.uniform(0, 1) < p_add_full_stop:
                        sentence += sentence[-1]
                elif (sentence[-1] == "!" and sentence[-2] != "!") or (
                    sentence[-1] == "！" and sentence[-2] != "！"
                ):
                    while random.uniform(0, 1) < p_add_exclamation_mark:
                        sentence += sentence[-1]
                elif (sentence[-1] == "?" and sentence[-2] != "?") or (
                    sentence[-1] == "？" and sentence[-2] != "？"
                ):
                    while random.uniform(0, 1) < p_add_question_mark:
                        sentence += sentence[-1]

                if sentence[-1].isalpha():
                    while random.uniform(0, 1) < p_add_semi_colon:
                        sentence += ";"

                corrupted_ud_dev_sentences.append(sentence)

            corrupted_ud_train_sentences = []
            for sentence in ud_train_sentences:

                if len(sentence) < 2:
                    continue

                if random.uniform(0, 1) < p_casing:
                    sentence = sentence[0].lower() + sentence[1:]

                if random.uniform(0, 1) < p_punct:
                    sentence = sentence.rstrip(".?!;。！？")

                if len(sentence) < 2:
                    continue

                if (sentence[-1] == "." and sentence[-2] != ".") or (
                    sentence[-1] == "。" and sentence[-2] != "。"
                ):
                    while random.uniform(0, 1) < p_add_full_stop:
                        sentence += sentence[-1]
                elif (sentence[-1] == "!" and sentence[-2] != "!") or (
                    sentence[-1] == "！" and sentence[-2] != "！"
                ):
                    while random.uniform(0, 1) < p_add_exclamation_mark:
                        sentence += sentence[-1]
                elif (sentence[-1] == "?" and sentence[-2] != "?") or (
                    sentence[-1] == "？" and sentence[-2] != "？"
                ):
                    while random.uniform(0, 1) < p_add_question_mark:
                        sentence += sentence[-1]

                if sentence[-1].isalpha():
                    while random.uniform(0, 1) < p_add_semi_colon:
                        sentence += ";"

                corrupted_ud_train_sentences.append(sentence)

            print(f"{lang_code}: {len(corrupted_ud_train_sentences)}")

            ud_data[lang_code]["train"] = corrupted_ud_train_sentences
            ud_data[lang_code]["dev"] = corrupted_ud_dev_sentences

    all_training_data = []

    for lang_code in langs:
        if "train" in ud_data[lang_code]:
            print(f"{lang_code}: {len(ud_data[lang_code]['train'])}")
            all_training_data.extend(ud_data[lang_code]["train"])

    all_dev_data = []

    for lang_code in langs:
        if "dev" in ud_data[lang_code]:
            all_dev_data.extend(ud_data[lang_code]["dev"])

    print("Number of training sentences:", len(all_training_data))
    print("Number of dev sentences:", len(all_dev_data))

    all_data = {"train": all_training_data, "dev": all_dev_data}

    with open(os.path.join(args.output_dir, "augmented_ud_training_data.pkl"), 'wb') as f:
        pickle.dump(all_data, f)
