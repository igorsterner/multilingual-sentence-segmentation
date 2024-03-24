"""Original file is located at
    https://colab.research.google.com/drive/1eqYurteJGqwqf6WZT8OARS6xfQ-sQiO3
"""

"""Then you need to install Git-LFS. Uncomment the following instructions:"""

"""Make sure your version of Transformers is at least 4.11.0 since the functionality was introduced in that version:"""


import pickle
from pathlib import Path

import numpy as np
import transformers
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

import wandb

print(transformers.__version__)

model_checkpoint = "xlm-roberta-base"
batch_size = 32

label2id = {
    "O": 0,
    "|": 1,
}

id2label = {v: k for k, v in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)


def tokenize_and_assign_labels(sentence):
    tokenized_inputs = tokenizer(sentence, truncation=True)
    tokenized_inputs["input_ids"] = tokenized_inputs["input_ids"][1:-1]
    labels = [0] * (len(tokenized_inputs["input_ids"]) - 1) + [1]
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


ud_data_path = "../data/preprocessed_training_data/augmented_ud_training_data.pkl"

with open(ud_data_path, "rb") as f:
    ud_data = pickle.load(f)

train_sentences = ud_data["train"]
dev_sentences = ud_data["dev"]

print(len(dev_sentences))
print(len(train_sentences))

print(len(set(train_sentences).intersection(set(dev_sentences))))

max_seq_length = 510


def pack_sentences_together(sentences, max_seq_length):
    packed_sentences = []
    packed_labels = []

    current_length = 0
    current_sentences = []
    current_labels = []

    for sentence in sentences:
        tokenized_sentence = tokenize_and_assign_labels(sentence)
        sentence_length = len(tokenized_sentence["input_ids"])

        # Check if adding the current sentence would exceed the max sequence length
        if current_length + sentence_length > max_seq_length:
            # Sentence can't be added without exceeding the max length, so pack what we have so far
            # Add a 0 at the beginnign of current_sentences and a 2 at the end
            current_sentences = [0] + current_sentences + [2]
            packed_sentences.append(current_sentences)

            current_labels = [0] + current_labels + [0]
            packed_labels.append(current_labels)
            # Reset for the next batch of sentences
            current_sentences = []
            current_labels = []
            current_length = 0

        # Add the current sentence to the batch
        current_sentences += tokenized_sentence["input_ids"]
        current_labels += tokenized_sentence["labels"]
        current_length += sentence_length

    return packed_sentences, packed_labels


packed_train_sentences, packed_train_labels = pack_sentences_together(
    train_sentences, max_seq_length
)
packed_dev_sentences, packed_dev_labels = pack_sentences_together(
    dev_sentences, max_seq_length
)

train_dataset = Dataset.from_dict(
    {"input_ids": packed_train_sentences, "labels": packed_train_labels}
)

dev_dataset = Dataset.from_dict(
    {"input_ids": packed_dev_sentences, "labels": packed_dev_labels}
)

print("Packed train dataset size:", len(train_dataset))
print("Packed test dataset size:", len(dev_dataset))


model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint, num_labels=len(label2id), id2label=id2label, label2id=label2id
)

model_name = model_checkpoint.split("/")[-1]
experiment_name = f"{model_name}-Multilingual-Sentence-Segmentation-v4"

run = wandb.init(project="Sentence segmentation", entity="igorsterner")
wandb.run.name = experiment_name

args = TrainingArguments(
    output_dir=Path("../data/models") / experiment_name,
    overwrite_output_dir=True,
    evaluation_strategy="steps",
    eval_steps=100,
    report_to="wandb",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    push_to_hub=True,
    hub_private_repo=True,
    save_total_limit=1,
    save_strategy="epoch",
    load_best_model_at_end=False,
)


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    flat_true_predictions = [item for sublist in true_predictions for item in sublist]

    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    flat_true_labels = [item for sublist in true_labels for item in sublist]

    precision, recall, f1, support = precision_recall_fscore_support(
        flat_true_predictions, flat_true_labels, pos_label="|", average="binary"
    )

    return {"precision": precision, "recall": recall, "f1": f1}


data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.evaluate()

trainer.push_to_hub()
