import sys

from flask import Flask, jsonify, request
import logging

logging.getLogger().setLevel(logging.INFO)

sys.path.append('../src')
from validate import Predictor

app = Flask(__name__)

char_vocab_path = '../artifacts/char_vocab.json'
lang_vocab_path = '../artifacts/lang_vocab.json'
model_path = '../artifacts/model'
params_path = '../artifacts/params.json'

predictor = Predictor(char_vocab_path, lang_vocab_path, params_path, model_path)


@app.route('/')
def index():
    welcome_msg = (
        'This is language identification API. <br>'
        'Read the docs here: <br>'
        'https://github.com/varvarvarvar/language_identification'
    )
    return welcome_msg


@app.route('/language_identification/api/v1.0/', methods=['GET'])
def identify_language():

    response = request.json

    if not response or 'text' not in response:
        error_msg = "Missing required argument 'text'."
        logging.error(error_msg)
        return jsonify(
            {
                'response': None,
                'text': None,
                'error': error_msg
            }
        ), 200

    text = response['text']
    prediction = predictor.predict([text])

    if 'error' in prediction:
        return jsonify(
            {
                'response': None,
                'text': text,
                'error': prediction['error']
            }
        ), 200

    logging.info(
        "Text: '%s', response: '%s'" % (text, prediction['response'])
    )
    return jsonify(
        {'response': prediction['response'][0], 'text': text}), 200
