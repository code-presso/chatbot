# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import os
import io
import re


def _read_data_file(file_path, train=True):
    sentences = []
    sentence = [[], [], []]
    for line in io.open(file_path, encoding="utf-8"):
        line = line.strip()
        if line == "":
            sentences.append(sentence)
            sentence = [[], [], []]
        else:
            idx, ejeol, ner_tag = line.split("\t")
            # idx는 0부터 시작하도록 
            sentence[0].append(int(idx))
            sentence[1].append(ejeol)
            if train:
                sentence[2].append(ner_tag)
            else:
                sentence[2].append("-")

    return sentences

def test_data_loader(root_path):
    # [ idx, ejeols, nemed_entitis ] each sentence
    file_path = os.path.join(root_path, 'test_data')

    return _read_data_file(file_path, False)


def data_loader(root_path):
    # [ idx, ejeols, nemed_entitis ] each sentence
    file_path = os.path.join(root_path, 'train_data')

    return _read_data_file(file_path)

def sentence_loader(input_sentence):

    sentences = []
    sentence = [[], [], []]

    re_com = re.compile("[!.?]$")
    match = re_com.search(input_sentence)

    if match:
        input_sentence = input_sentence[: match.end() - 1] + " " + input_sentence[match.end() - 1:]

    lst_ejeols = input_sentence.split(' ')

    for idx, ejeol in enumerate(lst_ejeols):
        sentence[0].append(idx)
        sentence[1].append(ejeol)
        sentence[2].append('-')

    sentences.append(sentence)

    return  sentences

if __name__ == "__main__":
    sentences = data_loader("data")
    print(sentences[0])
