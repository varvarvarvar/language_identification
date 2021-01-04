import sys

from flask import Flask, jsonify, request
import logging

logging.getLogger().setLevel(logging.INFO)

sys.path.append('src')
from validate import Predictor

app = Flask(__name__)

predictor = Predictor.default_predictor()
logging.info(predictor.keys())
logging.info(type(predictor['response']))

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

    global predictor

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

    # Reload predictor if no model was specified at startup
    if 'error' in predictor:
        predictor = Predictor.default_predictor()
        if 'error' in predictor:
            logging.info('Reloading model')
            return jsonify(
                {
                    'response': None,
                    'text': text,
                    'error': predictor['error']
                    }
                ), 200

    prediction = predictor['response'].predict([text])

    if 'error' in prediction:
        return jsonify(
            {
                'response': None,
                'text': text,
                'error': prediction['error']
            }
            ), 200

    logging.info(
        "Text: '%s', response: '%s'" % (text, prediction['response'][0])
    )
    return jsonify(
        {'response': prediction['response'][0], 'text': text}), 200
