# -*- coding: utf-8 -*-

from flask import Flask, request, abort

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
)

import requests
import argparse
import numpy as np

import ner_infer
import intent_infer


app = Flask(__name__)

NER_MODEL = None
INTENT_MODEL = None
INTENT_NUM_SEQ = 0
INTENT_INPUT_TOKEN = {}

FLAGS = None
ENTITIES = ['CVL_B', 'AFW_B']

# 라인 봇 api 정보 입력
line_bot_api = LineBotApi('YOUR_CHANNEL_ACCESS_TOKEN')
handler = WebhookHandler('YOUR_CHANNEL_SECRET')


@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    print("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    print("event: " + str(event))
    print("text: " + event.message.text)

    reply = ""

    # Call Infer function here
    intet_id = infer_intent(event.message.text)

    # 인사말 intent
    if intet_id == 0:
        reply = "안녕하세요 무었을 도와드릴까요?"
    # 구매 intent
    elif intet_id == 1:
        entities = infer_ner(event.message.text)

        if len(entities) == 0:
            reply = "구매하실 물건에 대해서 다시 말씀해 주세요."
        elif len(entities) == 1:
            reply = "{} 제품을 주문해도 될까요?."
        else:
            reply = "{} 제품들을 주문해도 될까요?"
    # 문의 intent
    elif intet_id == 2:
        reply = infer_generation(event.message.text)

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply))


def infer_intent(msg):
    # Infer Intent here
    print(msg)

    sentence = intent_infer.preprocess_sentence(msg, INTENT_NUM_SEQ, INTENT_INPUT_TOKEN)
    predictions = INTENT_MODEL.predict(sentence)
    intent_id = np.argmax(predictions, 1)[0]

    print(intent_id)

    return intent_id

def infer_generation(msg):
    # Infer KOSQUAD Model here
    print(msg)

    result = requests.get('http://localhost:5001/generate?sentences=' + msg)
    result = result.text

    print(result)

    return result


def infer_ner(msg):
    # Infer NER here
    print(msg)

    result = []

    ner_infer_result = NER_MODEL.predict(msg)

    for re in ner_infer_result:
        word = re[0]
        tag = re[1]
        print('{word} : {tag}'.format(word=word, tag=tag))

        if tag in ENTITIES:
            result.append(word)

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--ner_checkpoint_path',
        type=str,
        default='./ner/training_checkpoint/service',
        help='ner predict service checkpoint path'
    )
    parser.add_argument(
        '--intent_checkpoint_file',
        type=str,
        default='./intent/training_checkpoint/service/intent_model.h5',
        help='intent predict service checkpoint path'
    )

    FLAGS, umparsed = parser.parse_known_args()

    INTENT_MODEL, INTENT_NUM_SEQ, INTENT_INPUT_TOKEN = intent_infer.model_load(checkpoint_file=FLAGS.intent_checkpoint_file)
    NER_MODEL = ner_infer.Ner(checkpoint_dir=FLAGS.ner_checkpoint_path)

    # print(infer_intent("하이~"))
    # print(infer_ner("노란색 티셔츠를 주문해줘"))

    app.run()
