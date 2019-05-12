# Sense-specific word embeddings for Portuguese
Implementation of Sense-specific word embeddings for Portuguese

This repository consists of preprocessing and evaluation scripts used in the paper entitled Sense-specific word embeddings for Portuguese.
The preprocessing script cleaned corpora, tokenized and sentenced it.
Evaluation scripts can be used to measure the representativeness of a sense embedding model.

---

## About the paper

Paper can be read:

Trained embeddings models:
https://drive.google.com/open?id=1RROg4thS5Cj-gqByCEGMaaFxCJNlFZhD

### Abstract

Word embeddings are numerical vectors which can represent words or concepts in a low-dimensional continuous space. These vectors are able to capture useful syntactic and semantic information, such as regularities in natural language. Although very useful in many applications, the traditional approaches for generating word embeddings like Word2Vec, GloVe, Wang2Vec and FastText have a strict drawback: they produce a single vector representation for a given word ignoring the fact that ambiguous words can assume different meanings for which different vectors should be generated. This mixture of meanings can be a problem for several applications. For example, in a language understanding task, by using the embedding of an ambiguous word like the Portuguese word banco (bank), all possible meanings of it -- such as financial institution, blood bank, or an item of furniture -- will be mixed in a single numerical vector, causing a wrong semantic interpretation of the sentence in which it occurs. In this paper we present the first experiments carried out for generating sense-specific word embeddings for Portuguese, in which, instead of word occurrences, word senses are represented in sense vectors. Our experiments show that sense vectors outperform traditional word vectors in syntactic and semantic tasks, proving that the language resource generated here can improve the performance of NLP tasks in Portuguese.

---

### Contents

* [Installation](#installation)
* [Usage](#usage)
  * [Preprocessing text file](#preprocessing-text-file)
  * [Syntactic and Semantic analogies evaluation](#syntactic-and-semantic-analogies-evaluation)
  * [Semantic Similarity evaluation](#semantic-similarity-evaluation)
  * [Word Sense Disambiguation evaluation](#word-sense-disambiguation-evaluation)

---

## Installation
```
virtualenv venv -p python3
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download pt
```

## Usage

### Trained embeddings models

Download the pre-trained sense vectors and add them to the models folder.

### Preprocessing text file (in order to train embedding models)

Script used for cleaning corpus

All emails are mapped to a EMAIL token.
All numbers are mapped to 0 token.
All urls are mapped to URL token.
Different quotes are standardized.
Different hiphen are standardized.
HTML strings are removed.
All text between brackets are removed.
All sentences shorter than 5 tokens were removed.
```
python preprocessing.py <input_file.txt> <output_file.txt>
```

Annotate the corpus with PoS tags with the nlpnet tool
```
python postagging.py <input_folder.txt> <output_folder.txt>
```

### Syntactic and Semantic analogies evaluation

This method is similar to that one developed by [nlx-group](https://github.com/nlx-group/lx-dsemvectors)
```
python evaluate.py <embedding_model.txt> <testset.txt>
```
#### Brazilian Portuguese testsets

Only syntactic analogies
```
python evaluate.py <embedding_model.txt> analogies/testset/LX-4WAnalogiesBr_syntactic.txt
```
Only semantic analogies
```
python evaluate.py <embedding_model.txt> analogies/testset/LX-4WAnalogiesBr_semantic.txt
```
All analogies
```
python evaluate.py <embedding_model.txt> analogies/testset/LX-4WAnalogiesBr.txt
```
#### European Portuguese testsets

Only syntactic analogies
```
python evaluate.py <embedding_model.txt> analogies/testset/LX-4WAnalogies_syntactic.txt
```
Only semantic analogies
```
python evaluate.py <embedding_model.txt> analogies/testset/LX-4WAnalogies_semantic.txt
```
All analogies
```
python evaluate.py <embedding_model.txt> analogies/testset/LX-4WAnalogies.txt
```

### Semantic Similarity evaluation

Sentence Similarity
```
python evaluate.py <embedding_model.txt> --lang
```
Parameter **--lang** can be set depending on portuguese variant chosen.

Brazilian Portuguese
```
br
```
European Portuguese
```
pt
```

### Word Sense Disambiguation evaluation
```
python lexical_sample.py
```

