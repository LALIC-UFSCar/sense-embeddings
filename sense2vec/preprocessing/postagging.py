#!/usr/bin/env python
# coding: utf-8
"""This script can be used to preprocess a corpus for training a sense2vec
model. It take text file with one sentence per line, and outputs a text file
with one sentence per line in the expected sense2vec format (merged noun
phrases, concatenated phrases with underscores and added "senses").

Example input:
Rats, mould and broken furniture: the scandal of the UK's refugee housing

Example output:
Rats|NOUN ,|PUNCT mould|NOUN and|CCONJ broken_furniture|NOUN :|PUNCT
the|DET scandal|NOUN of|ADP the|DET UK|GPE 's|PART refugee_housing|NOUN

DISCLAIMER: The sense2vec training and preprocessing tools are still a work in
progress. Please note that this script hasn't been optimised for efficiency yet
and doesn't paralellize or batch up any of the work, so you might have to
add this functionality yourself for now.
"""
from __future__ import print_function, unicode_literals

import spacy
import nlpnet
from pathlib import Path
from tqdm import tqdm
import plac
import glob, os


def represent_word_nlpnet(word, tagger):
    tag = tagger.tag(str(word))[0][0][1]
    return str(word) + '|' + tag


def represent_sentence_nlpnet(sent, tagger):
    tags = tagger.tag(str(sent))
    sent_tagged = ""
    for tag in tags[0]:
        word = str(tag).replace("(", "")
        word = word.replace(")", "")
        word = word.replace("\', ", "|")
        word = word.replace("\'", "")
        sent_tagged = sent_tagged + ' ' + word
    return sent_tagged


def represent_doc(doc):
    tagger = nlpnet.POSTagger('./pos-pt', language='pt')
    strings = []
    for sent in doc.sents:
        if sent.text.strip():
            sentence = represent_sentence_nlpnet(sent, tagger)
            strings.append(sentence)
    return '\n'.join(strings) + '\n' if strings else ''


@plac.annotations(
    in_file=("Path to input file", "positional", None, str),
    out_file=("Path to output file", "positional", None, str),
    spacy_model=("Name of spaCy model to use", "positional", None, str),
    n_workers=("Number of workers", "option", "n", int))
def main(in_file, out_file, spacy_model='pt_core_news_sm', n_workers=4): #en_core_web_sm
    input_path = str(Path(in_file))
    print(input_path)
    output_path = str(Path(out_file))
    print(output_path)
    #if not input_path.exists():
        #raise IOError("Can't find input file: {}".format(input_path))
    nlp = spacy.load(spacy_model)
    print("Using spaCy model {}".format(spacy_model))
    lines_count = 0

    os.chdir(input_path)
    for filename in glob.iglob('*.txt'):
        print("corpus: ", filename)
        input_path = Path(filename)
        with input_path.open('r', encoding='utf8') as texts:
            docs = nlp.pipe(texts, n_threads=n_workers)
            lines_pos = (represent_doc(doc) for doc in docs)
            output_path = Path(output_path + filename + "_pos")
            with output_path.open('w', encoding='utf8') as f:
                for line in tqdm(lines_pos, desc='Lines', unit=''):
                    lines_count += 1
                    f.write(line)
    print("Successfully preprocessed {} lines".format(lines_count))
    print("{}".format(output_path.resolve()))


if __name__ == '__main__':
    plac.call(main)

#python postagging.py "../../corpora" "../../corpora"
