# -*- coding: utf-8 -*-

import json
import numpy as np

from tensorflow.keras import models


def model_load(checkpoint_file):

    model = models.load_model(checkpoint_file)

    # 입력데이터 전처리에 필요한 속성 정보
    json_data = open('./intent/model_parameter.json').read()
    model_parameter = json.loads(json_data)
    num_sequence = int(model_parameter['num_sequence'])
    json_data = open('./intent/input_token_index.json').read()
    input_token_index = json.loads(json_data)

    print('intent modle loaed')

    return model, num_sequence, input_token_index

def preprocess_sentence(sentence, num_sequence, input_token_index):

    input_data = np.zeros((1, num_sequence), dtype='float32')

    for t, char in enumerate(sentence):
        if t < num_sequence:
            input_data[0, t] = input_token_index[char]

    return input_data