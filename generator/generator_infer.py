# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import tensorflow as tf

tf.enable_eager_execution()

import json

from generator_model import Encoder, Decoder
import preprocess as ps


class Generator(tf.keras.Model):
    def __init__(self, checkpoint_dir='./asset/training_checkpoint/service'):
        super(Generator, self).__init__()
        # 모델 구현에 필요한 속성 정보(학습데이터 분석을 통해 얻어낸 모델의 하이퍼파라미터 정보)
        json_data = open('./asset/model_parameter.json').read()
        self.model_parameter = json.loads(json_data)
        self.encoder_seq_length = int(self.model_parameter['encoder_seq_length'])
        self.decoder_seq_length = int(self.model_parameter['decoder_seq_length'])
        self.embedding_dim      = int(self.model_parameter['embedding_dim'])
        self.units              = int(self.model_parameter['units'])

        json_data = open('./asset/input_token_index.json').read()
        self.input_token_index = json.loads(json_data)

        json_data = open('./asset/target_token_index.json').read()
        self.target_token_index = json.loads(json_data)

        json_data = open('./asset/reverse_target_word_index.json').read()
        self.reverse_target_word_index = json.loads(json_data)

        self.vocab_input_size = len(self.input_token_index)
        self.vocab_tar_size = len(self.target_token_index)

        # 모델 구현에 필요한 속성 정보
        self.hidden = [tf.zeros((1, self.units))]
        self.encoder = Encoder(self.vocab_input_size, self.embedding_dim, self.units, batch_sz=1)
        self.decoder = Decoder(self.vocab_tar_size, self.embedding_dim, self.units, batch_sz=1)

        checkpoint = tf.train.Checkpoint(encoder=self.encoder,
                                         decoder=self.decoder)

        # 서비스를 위한 학습 결과물(check point) 정보를 불러와 모델에 load
        # 학습 결과물 폴더 내(./asset/training_checkpoint_cpu/service) checkpoint 파일에서
        # 최신 체크포인트에 대한 정보 확인
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

        print('generator model loaded')


    def predict(self, x):
        # print('generator_input: {}'.format(x))

        sentence = ps.normalizeString(x)
        inputs = [self.input_token_index[char] for t, char in enumerate(sentence)]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=self.encoder_seq_length, padding='post')
        inputs = tf.convert_to_tensor(inputs)

        result = ''

        hidden = [tf.zeros((1, self.units))]
        enc_out, enc_hidden = self.encoder(inputs, hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([self.target_token_index['\t']], 0)

        for t in range(self.decoder_seq_length):
            predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_out)

            predicted_id = tf.argmax(predictions[0]).numpy()
            predicted_word = self.reverse_target_word_index.get(str(predicted_id), "")

            result += predicted_word

            if predicted_word == '\n':
                return result

            # the predicted ID is fed back into the model
            dec_input = tf.expand_dims([predicted_id], 0)

        return result
