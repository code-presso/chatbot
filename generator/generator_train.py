# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import argparse
import tensorflow as tf

tf.enable_eager_execution()


import numpy as np
import os
import time
import matplotlib.pyplot as plt
import json
import sys

from generator_model import Encoder, Decoder
import preprocess as ps


FLAGS = None

INPUT_TOKEN_INDEX = {}
TARGET_TOKEN_INDEX = {}
TARGET_INDEX_TOKEN = {}
MODEL_PARAMETER = {}

def loss_function(real, pred):

  mask = 1 - np.equal(real, 0)
  loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask

  return tf.reduce_mean(loss_)


def generate(sentence, encoder, decoder, max_length_inp, max_length_targ):

    attention_plot = np.zeros((max_length_targ, max_length_inp))

    inputs = [INPUT_TOKEN_INDEX[char] for t, char in enumerate(sentence)]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, FLAGS.units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([TARGET_TOKEN_INDEX['\t']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)

        # storing the attention weigths to plot later on
        attention_weights = tf.reshape(attention_weights, (-1,))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += TARGET_INDEX_TOKEN[predicted_id]

        if TARGET_INDEX_TOKEN[predicted_id] == '\n':
            return result, sentence, attention_plot

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot


def plot_attention(attention, sentence, predicted_sentence):
    # plt.register_cmap(name='viridis', cmap=cmaps.viridis)
    # plt.set_cmap(cmaps.viridis)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    plt.savefig(os.path.join(FLAGS.attention_img_path, str(int(time.time()))), bbox_inches='tight')


def evaluate(sentence, encoder, decoder, max_length_inp, max_length_targ):

    result, sentence, attention_plot = generate(sentence, encoder, decoder, max_length_inp, max_length_targ)

    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(result))

    attention_plot = attention_plot[:len(result), :len(sentence)]
    plot_attention(attention_plot, list(sentence), list(result))


def save_word_analysis_data():

    with open('./asset/input_token_index.json', 'w') as fp:
        json.dump(INPUT_TOKEN_INDEX, fp)
    with open('./asset/target_token_index.json', 'w') as fp:
        json.dump(TARGET_TOKEN_INDEX, fp)
    with open('./asset/reverse_target_word_index.json', 'w') as fp:
        json.dump(TARGET_INDEX_TOKEN, fp)
    with open('./asset/model_parameter.json', 'w') as fp:
        json.dump(MODEL_PARAMETER, fp)

def train (n_batch, dataset, decoder, encoder, optimizer, checkpoint):

    checkpoint_prefix = os.path.join(FLAGS.checkpoint_path, 'ckpt')

    for epoch in range(FLAGS.epochs):
        start = time.time()

        hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(dataset):
            loss = 0

            with tf.GradientTape() as tape:
                enc_output, enc_hidden = encoder(inp, hidden)

                dec_hidden = enc_hidden

                dec_input = tf.expand_dims([TARGET_TOKEN_INDEX['\t']] * FLAGS.batch_size, 1)

                # Teacher forcing - feeding the target as the next input
                for t in range(0, targ.shape[1]):
                    # passing enc_output to the decoder
                    predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
                    loss += loss_function(targ[:, t], predictions)

                    # using teacher forcing
                    dec_input = tf.expand_dims(targ[:, t], 1)

            batch_loss = (loss / int(targ.shape[1]))

            total_loss += batch_loss

            variables = encoder.variables + decoder.variables

            gradients = tape.gradient(loss, variables)

            optimizer.apply_gradients(zip(gradients, variables))

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))
        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / n_batch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


def main(_):

    global INPUT_TOKEN_INDEX, TARGET_TOKEN_INDEX, TARGET_INDEX_TOKEN, MODEL_PARAMETER

    input_tensor_train, target_tensor_train, input_tensor_val, \
    max_encoder_seq_length, max_decoder_seq_length, \
    INPUT_TOKEN_INDEX, TARGET_TOKEN_INDEX, \
    reverse_input_word_index, TARGET_INDEX_TOKEN = ps.preprocess_data(FLAGS.num_samples, FLAGS.data_path)

    buffer_size = len(input_tensor_train)
    n_batch = buffer_size // FLAGS.batch_size

    dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(buffer_size)
    dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)

    encoder = Encoder(len(INPUT_TOKEN_INDEX), FLAGS.embedding_dim, FLAGS.units, FLAGS.batch_size)
    decoder = Decoder(len(TARGET_TOKEN_INDEX), FLAGS.embedding_dim, FLAGS.units, FLAGS.batch_size)

    optimizer = tf.train.AdamOptimizer()

    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)

    train(n_batch, dataset, decoder, encoder, optimizer, checkpoint)

    MODEL_PARAMETER = {
        "encoder_seq_length": max_encoder_seq_length,
        "decoder_seq_length": max_decoder_seq_length,
        "embedding_dim": FLAGS.embedding_dim,
        "units": FLAGS.units,
    }

    save_word_analysis_data()

    # restoring the latest checkpoint in checkpoint_dir
    checkpoint.restore(tf.train.latest_checkpoint(FLAGS.checkpoint_path))

    for val in input_tensor_val[:10]:
        sentence = ''.join([reverse_input_word_index[id] for id in val])
        evaluate(sentence, encoder, decoder, max_encoder_seq_length, max_decoder_seq_length)

    for val in input_tensor_train[:10]:
        sentence = ''.join([reverse_input_word_index[id] for id in val])
        evaluate(sentence, encoder, decoder, max_encoder_seq_length, max_decoder_seq_length)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--num_samples',
        type=int,
        default=50,
        help='training samples max count'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='./data/KorQuAD_v1.0_train.csv',
        help='training data path'
    )
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='./asset/training_checkpoint/' + str(int(time.time())),
        help='checkpoint file path'
    )
    parser.add_argument(
        '--attention_img_path',
        type=str,
        default='./asset/attention_img',
        help='evaluate attention img file path'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=2,
        help='전체 학습 데이터 소진 횟수'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='학습 시 한번에 사용되는 데이터의 건수'
    )
    parser.add_argument(
        '--embedding_dim',
        type=int,
        default=512,
        help='evaluate attention img file path'
    )
    parser.add_argument(
        '--units',
        type=int,
        default=1024,
        help='인코더 디코더 모델의 unit 사이즈'
    )

    FLAGS, unparsed = parser.parse_known_args()

    main([sys.argv[0]] + unparsed)