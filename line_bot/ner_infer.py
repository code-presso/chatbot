#-*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import os

from ner_model import Model
from ner_dataset_batch import Dataset
from ner_data_loader import sentence_loader


class Ner():
    def __init__(self, checkpoint_dir='./ner/training_checkpoint/service'):
        
        self.prarameter = {
            'mode': 'infer',                              # Operation mode
            'necessary_file': "./ner/necessary.pkl",      # Train output, analysis data result
            'train_lines': 1,                             # Maximum train lines(in infer case = 1)
            'learning_rate': 0.02,                        # Learning rate
            'keep_prob': 0.65,                            # Dropout_rate
            'word_embedding_size': 16,                    # Word, WordPos Embedding Size
            'char_embedding_size': 16,                    # Char Embedding Size
            'tag_embedding_size': 16,                     # Tag Embedding Size
            'lstm_units': 16,                             # Hidden unit size
            'char_lstm_units': 32,                        # Hidden unit size for Char rnn
            'sentence_length': 180,                       # Maximum words in sentence
            'word_length': 8                              # Maximum chars in word
        }

        extern_data = []

        self.dataset = Dataset(self.prarameter, extern_data)
        self.dic_idx_tag = {v: k for k, v in self.dataset.necessary_data['ner_tag'].items() }

        self.model = Model(self.dataset.parameter)
        self.model.build_model()

        self.sess = tf.Session()
        self.saver = tf.train.Saver()

        last_ckpt_path = tf.train.latest_checkpoint(checkpoint_dir)
        self.saver.restore(self.sess, last_ckpt_path)
        # self.saver.restore(self.sess, os.path.join(checkpoint_dir, 'ner_model'))

        print('ner model loaded')

    def predict(self, input_sentence):

        # 데이터 전처리
        extern_data = sentence_loader(input_sentence)
        self.dataset.make_input_data(extern_data)

        # 테스트 셋을 측정한다.
        for morph, ne_dict, character, seq_len, char_len, _, step \
                in self.dataset.get_data_batch_size(self.prarameter['train_lines'], False):
            feed_dict = { self.model.morph : morph,
                          self.model.ne_dict : ne_dict,
                          self.model.character : character,
                          self.model.sequence : seq_len,
                          self.model.character_len : char_len,
                          self.model.dropout_rate : 1.0
                        }

            prediction = self.sess.run(self.model.viterbi_sequence, feed_dict=feed_dict)

        par_sentence = extern_data[0][1]
        prediction = [self.dic_idx_tag[prediction[0][idx]] for idx in range(len(par_sentence))]

        return zip(par_sentence, prediction)