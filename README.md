# Multilingual Sentence Segmentation

## Why?

Sentence segmentation is an early step in many NLP pipelines. But segmenters vary in performance massively across (1) language and (2) domain (e.g. formal vs. social media text). 

This is a multilingual model designed to sentence segment text of many different languages. It is not language or domain specific. This model is particularly good at segmenting poorly punctuated text, such as text from social media.

## How?

1. Start with sentence segmented training corpora in many difference languages (from Universal Dependencies)
2. Corrupt the training corpus with heuristics: lowercase the first letter, remove trailing punctuation, repeat trailing punctuation (see src/extract_augmented_ud_training_data.py for the heuristics we used)
3. Finetune a multilingual language model to classify each token as the end of a sentence, or not
4. Evaluate on sentence segmentation from a range of sources of varying domains (e.g. code-switching data, mt data, ud data (without corruption this time, of course))

## Is it any good?

Results are given below! We compare against the largest WtP Split model (`wtp-canine-s-12l-no-adapters`) and multilingual segmentation from spacy (`xx_sent_ud_sm`). We are very often better. None of the models is told what language the text is.

## Credit

Most the code in this repo has been adapted from https://github.com/bminixhofer/wtpsplit [1], especially the data extraction. Full credit is given to Benjamin Minixhofer, Jonas Pfeiffer and Ivan Vulić for that work!

# Usage

The simplest way to use the system is with a Huggingface pipeline. You can try out our model here: https://huggingface.co/igorsterner/xlmr-multilingual-sentence-segmentation

```
from transformers import pipeline

pipe = pipeline(
    "token-classification",
    model="igorsterner/xlmr-multilingual-sentence-segmentation",
) 
```

If your text is longer than the context window, add stride=5. For single-GPU speedups, add device=0.

Segment and process:

```
text = "" # add your text here
output = pipe(text)

sentences = []
start = 0
for token in output:
    sentence = text[start : token["end"]]
    sentences.append(sentence)
    start = token["end"]

if start < len(text):
    sentences.append(text[start:])
```

# Results

All results here are percentage F1

## Opus100 [2]

Who wins most? XLM-RoBERTa: 56, WtPSplit: 12, Spacy (multilingual): 8


|                      | af        | am        | ar        | az        | be        | bg        | bn        | ca        | cs        | cy        | da        | de        | el        | en        | eo        | es        | et        | eu        | fa        | fi        | fr        | fy        | ga        | gd        | gl        | gu        | ha        | he        | hi        | hu        | hy        | id        | is        | it        | ja        | ka        | kk        | km        | kn        | ko        | ku        | ky        | lt        | lv        | mg        | mk        | ml        | mn        | mr        | ms        | my        | ne        | nl        | pa        | pl        | ps        | pt        | ro        | ru        | si        | sk        | sl        | sq        | sr        | sv        | ta        | te        | th        | tr        | uk        | ur        | uz        | vi        | xh        | yi        | zh        |
|:---------------------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|
| Spacy (multilingual) | 42.61     | 6.69      | 58.52     | 73.59     | 34.78     | 93.74     | 38.04     | 88.76     | 87.70     | 26.30     | 90.52     | 74.15     | 89.75     | 89.25     | 88.77     | 90.95     | 87.26     | 81.20     | 55.40     | 93.28     | 85.77     | 21.49     | 60.61     | 36.83     | 88.77     | 5.59      | **89.39** | **92.21** | 53.33     | 93.26     | 24.14     | 90.13     | **95.38** | 86.32     | 0.20      | 38.24     | 42.39     | 0.10      | 9.66      | 51.79     | 27.64     | 21.77     | 76.91     | 77.02     | 83.60     | **93.74** | 39.09     | 33.23     | 86.56     | 87.39     | 0.10      | 6.59      | **93.65** | 5.26      | 92.42     | 2.41      | 92.07     | 91.63     | 75.95     | 75.91     | 92.13     | 93.00     | **92.96** | **95.01** | 93.52     | 36.97     | 64.59     | 21.64     | **94.05** | 89.68     | 29.17     | 64.99     | 90.59     | 64.89     | 4.14      | 0.09      |
| WtPSplit             | 76.90     | **59.08** | 68.08     | 76.42     | 71.29     | 93.97     | 79.76     | 89.79     | 89.36     | 73.21     | 90.02     | 80.74     | 92.80     | 91.91     | 92.24     | 92.11     | 84.47     | 87.24     | 59.97     | 91.96     | 88.53     | 65.84     | 79.49     | 83.33     | 90.31     | **70.51** | 82.43     | 90.58     | 66.70     | 93.00     | 87.14     | 89.80     | 94.77     | 87.43     | **41.79** | **91.26** | 73.25     | **69.54** | 68.98     | 56.21     | **79.12** | 83.94     | 81.33     | 82.70     | **89.33** | 92.87     | 80.81     | 73.26     | 89.20     | 88.51     | **65.54** | **71.33** | 92.63     | 64.11     | 92.72     | **62.84** | 91.05     | 90.91     | 84.23     | 80.32     | 92.30     | 92.19     | 90.32     | 94.76     | 92.08     | 63.48     | 76.49     | 68.88     | 93.30     | 89.60     | 52.59     | **77.79** | 91.29     | 80.28     | **75.70** | 71.64     |
| XLM-RoBERTa (ours)   | **83.97** | 41.59     | **81.56** | **81.30** | **85.68** | **94.34** | **84.10** | **91.80** | **91.23** | **78.72** | **92.64** | **86.73** | **93.87** | **94.50** | **94.57** | **93.18** | **90.19** | **90.28** | **74.79** | **94.06** | **90.46** | **81.76** | **84.33** | **85.62** | **92.55** | 67.26     | 86.61     | 91.22     | **72.69** | **94.53** | **89.83** | **92.24** | 93.78     | **89.27** | 41.43     | 78.39     | **89.15** | 36.60     | **70.51** | **82.77** | 58.14     | **89.41** | **89.99** | **88.25** | 86.82     | 92.81     | **86.14** | **94.73** | **93.25** | **92.44** | 49.39     | 66.02     | 93.60     | **69.22** | **93.51** | 61.86     | **92.84** | **93.19** | **89.47** | **86.24** | **92.95** | **93.46** | 91.79     | 94.16     | **93.93** | **72.74** | **81.77** | **74.49** | 93.17     | **92.15** | **62.92** | 75.65     | **93.41** | **84.89** | 56.85     | **77.07** |


## Universal Dependencies [3]

Who wins most? XLM-RoBERTa: 24, WtPSplit: 17 Spacy (multilingual): 13


|                      | af        | ar        | be        | bg        | bn        | ca        | cs        | cy        | da        | de        | el        | en        | es        | et        | eu        | fa        | fi        | fr        | ga        | gd        | gl        | he        | hi        | hu        | hy        | id        | is        | it        | ja        | jv        | kk        | ko        | la        | lt        | lv        | mr        | nl        | pl        | pt        | ro        | ru        | sk        | sl        | sq         | sr        | sv        | ta        | th        | tr        | uk        | ur        | vi        | zh        |
|:---------------------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:-----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|
| Spacy (multilingual) | **98.47** | 80.38     | 80.27     | 93.62     | 51.85     | **98.95** | 89.68     | 98.89     | 94.96     | 88.02     | 94.16     | 92.20     | **98.70** | 93.77     | 95.79     | **99.83** | 92.88     | 96.33     | **96.67** | 63.04     | 92.37     | 94.37     | 0.32      | **98.45** | 11.39     | 98.01     | **95.41** | 92.49     | 0.37      | 98.03     | 96.21     | **99.80** | 0.09      | 93.86     | **98.52** | 92.13     | 92.86     | 97.02     | 94.91     | **98.05** | 84.31     | 90.26     | **98.23** | **100.00** | 97.84     | 94.91     | 66.67     | 1.95      | **97.63** | 94.16     | 0.37      | 96.40     | 0.40      |
| WtPSplit             | 98.27     | **83.00** | 89.28     | **98.16** | **99.12** | 98.52     | 92.98     | **99.26** | 94.56     | 96.13     | **96.94** | 94.73     | 97.60     | 94.09     | 97.24     | 97.29     | 94.69     | **96.71** | 86.60     | 72.17     | **98.87** | 95.79     | 96.78     | 96.08     | **96.80** | **98.41** | 86.39     | 95.45     | **95.84** | **98.18** | 96.28     | 99.11     | 91.43     | **97.67** | 96.42     | 91.84     | 93.61     | 95.92     | **96.13** | 81.50     | 86.28     | 95.57     | 96.85     | 99.17      | **98.45** | **95.86** | **97.54** | 70.26     | 96.00     | 92.08     | 93.79     | 92.97     | **97.25** |
| XLM-RoBERTa (ours)   | 96.81     | 78.99     | **91.60** | 97.89     | **99.12** | 95.99     | **96.05** | 97.17     | **96.62** | **96.29** | 94.33     | **94.76** | 95.73     | **96.20** | **97.37** | 97.49     | **96.34** | 95.70     | 89.78     | **84.20** | 95.72     | **95.95** | **97.51** | 96.24     | 95.62     | 97.22     | 92.93     | **96.88** | 94.23     | 96.29     | **98.40** | 97.46     | **96.35** | 95.82     | 96.91     | **95.92** | **96.27** | **97.24** | 95.83     | 94.63     | **91.59** | **95.88** | 96.43     | 98.36      | 96.83     | 94.95     | 95.93     | **89.26** | 96.52     | **94.59** | **96.20** | **97.31** | 95.12     |

## Ersatz [4]

Who wins most? XLM-RoBERTa: 10, WtPSplit: 8, Spacy (multilingual): 4


|                      | ar        | cs        | de        | en        | es        | et        | fi        | fr        | gu        | hi        | ja        | kk        | km        | lt        | lv        | pl        | ps        | ro        | ru        | ta        | tr        | zh        |
|:---------------------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|
| Spacy (multilingual) | **91.26** | 96.46     | 93.89     | 94.40     | 97.31     | **97.15** | 94.99     | 96.43     | 4.44      | 18.41     | 0.18      | 97.11     | 0.08      | 93.53     | **98.73** | 93.69     | **94.44** | 94.87     | 93.45     | 68.65     | 95.39     | 0.10      |
| WtPSplit             | 89.45     | 93.41     | 95.93     | **97.16** | **98.74** | 95.84     | 97.10     | **97.61** | 90.62     | 94.87     | **82.14** | 95.94     | **82.89** | **96.74** | 97.22     | 95.16     | 86.99     | **97.55** | **97.82** | 94.76     | 93.53     | 89.02     |
| XLM-RoBERTa (ours)   | 79.78     | **96.94** | **97.02** | 96.10     | 97.06     | 96.80     | **97.67** | 96.33     | **93.73** | **95.34** | 77.54     | **97.28** | 78.94     | 96.13     | 96.45     | **96.71** | 92.33     | 96.24     | 97.15     | **95.94** | **95.76** | **90.11** |

## German--English code-switching [5]

|                      | de        |
|:---------------------|:----------|
| Spacy (multilingual) | 79.55     |
| WtPSplit             | 77.41     |
| XLM-RoBERTa (ours)   | **85.78** |

[1] [Where’s the Point? Self-Supervised Multilingual Punctuation-Agnostic Sentence Segmentation](https://aclanthology.org/2023.acl-long.398) (Minixhofer et al., ACL 2023)

[2] [Improving Massively Multilingual Neural Machine Translation and Zero-Shot Translation](https://aclanthology.org/2020.acl-main.148) (Zhang et al., ACL 2020)

[3] [Universal Dependencies](https://aclanthology.org/2021.cl-2.11) (de Marneffe et al., CL 2021)

[4] [A unified approach to sentence segmentation of punctuated text in many languages](https://aclanthology.org/2021.acl-long.309) (Wicks & Post, ACL-IJCNLP 2021)

[5] [The Denglisch Corpus of German-English Code-Switching](https://aclanthology.org/2023.sigtyp-1.5) (Osmelak & Wintner, SIGTYP 2023)


# Replication

## Setup

```
conda create -n sentence-segmentation-env python=3.11
conda activate sentence-segmentation-env

pip install torch --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

## Generate training data and compile evaluation data

```
cd src
python extract_augmented_ud_training_data.py
python extract_eval_data.py
```

## Finetune XLM-RoBERTa

```
python train.py
```

## Evaluate

```
python eval.py
```
