# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
import re
from sklearn.model_selection import train_test_split


def normalizeString(s):
    hangul = re.compile(u'[^ ㄱ-ㅣ가-힣 ^☆; ^a-zA-Z.!?]+')
    match = hangul.search(s)

    result = []

    if not match:
        result = hangul.sub('', s)

    return result

def preprocess_data(num_samples, data_path):

    input_texts = []
    target_texts = []
    input_chars = set()
    target_chars = set()

    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')

    for line in lines[: min(num_samples, len(lines) - 1)]:
        # for line in lines:
        tmp_text = line.split('\t')

        if len(tmp_text) > 1:
            input_text = normalizeString(tmp_text[0])
            target_text = normalizeString(tmp_text[1])

        # "tab"을 목표 데이터의 시작, 종료 문자로 지정
        if len(input_text) > 0 and len(target_text) > 0:
            target_text = '\t' + target_text + '\n'
            input_texts.append(input_text)
            target_texts.append(target_text)
            for char in input_text:
                if char not in input_chars:
                    input_chars.add(char)
            for char in target_text:
                if char not in target_chars:
                    target_chars.add(char)

    input_chars = sorted(list(input_chars))
    target_chars = sorted(list(target_chars))
    num_encoder_tokens = len(input_chars)
    num_decoder_tokens = len(target_chars)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    print('Number of samples:', len(input_texts))
    print('Number of unique input tokens:', num_encoder_tokens)
    print('Number of unique output tokens:', num_decoder_tokens)
    print('Max sequence length for inputs:', max_encoder_seq_length)
    print('Max sequence length for outputs:', max_decoder_seq_length)

    input_token_index = dict(
        [(char, i) for i, char in enumerate(input_chars)])
    target_token_index = dict(
        [(char, i) for i, char in enumerate(target_chars)])

    reverse_input_word_index = dict(
        (i, char) for char, i in input_token_index.items())
    reverse_target_word_index = dict(
        (i, char) for char, i in target_token_index.items())

    encoder_input_data = np.zeros(
        (len(input_texts), max_encoder_seq_length),
        dtype='float32')
    decoder_input_data = np.zeros(
        (len(input_texts), max_decoder_seq_length),
        dtype='float32')
    decoder_target_data = np.zeros(
        (len(input_texts), max_decoder_seq_length),
        dtype='float32')

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t] = input_token_index[char]
        for t, char in enumerate(target_text):
            decoder_input_data[i, t] = target_token_index[char]
            if t > 0:
                # 디코더의 목표 데이터는 디코더 입력 데이터 보다 한 step 만큼 앞서 있음
                # 또한 디코더의 목표 데이터는 시작 문자(\t) 가 존재하지 않음
                decoder_target_data[i, t - 1] = target_token_index[char]

    # loss 함수 시 사용하는  softmax function(sparse_softmax_cross_entropy_with_logits) 에
    # label 데이터 입력을 위한 형 변환
    decoder_target_data = decoder_target_data.astype(np.int32)

    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(encoder_input_data,
                                                                                                    decoder_target_data,
                                                                                                    test_size=0.2)

    return input_tensor_train, target_tensor_train, input_tensor_val, \
            max_encoder_seq_length, max_decoder_seq_length, \
            input_token_index, target_token_index, \
            reverse_input_word_index, reverse_target_word_index