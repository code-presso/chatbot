# -*- coding: utf-8 -*-

from flask import Flask, jsonify, request
import requests
import argparse
import os

import generator_infer

app = Flask(__name__)

GENERATOR_MODEL = None
FLAGS = None

@app.route("/generate")
def generate():
    query = request.args.get('sentence')

    print('query  {}'.format(query))

    gen_sentence = GENERATOR_MODEL.predict(x=str(query))

    result = {'result': gen_sentence}

    print('rsult: {}'.format(result))

    return jsonify(result)


@app.route("/test")
def test():
    result = requests.get('http://localhost:5001/generate?q=hihihi')
    response = {'result': result.text}
    return jsonify(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--generator_checkpoint_path',
        type=str,
        default='./asset/training_checkpoint/service',
        help='generator service checkpoint path'
    )

    FLAGS, unparsed = parser.parse_known_args()

    GENERATOR_MODEL = generator_infer.Generator(
        checkpoint_dir=FLAGS.generator_checkpoint_path
    )

    # print(GENERATOR_MODEL.predict(x='대한민국 대통령은?'))

    app.run(host='0.0.0.0', port=5001, debug=False)