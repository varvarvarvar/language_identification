from flask import Flask, jsonify, request
import logging

#from .moesif_monitoring import moesif_settings
#from moesifwsgi import MoesifMiddleware

logging.getLogger().setLevel(logging.INFO)

import sys
sys.path.append('../src')
from validate import Predictor

app = Flask(__name__)

char_vocab_path = '../mlruns/0/e9b1c8c553904e19b1a7b3564a1ee8cf/artifacts/char_vocab.json'
lang_vocab_path = '../mlruns/0/e9b1c8c553904e19b1a7b3564a1ee8cf/artifacts/lang_vocab.json'
model_path = '../mlruns/0/e9b1c8c553904e19b1a7b3564a1ee8cf/artifacts/49.model'
params_path = '../mlruns/0/e9b1c8c553904e19b1a7b3564a1ee8cf/artifacts/params.json'

predictor = Predictor(char_vocab_path, lang_vocab_path, params_path, model_path)

# app.wsgi_app = MoesifMiddleware(app.wsgi_app, moesif_settings)

# texts = ['Savannklimat råder i trakten. Årsmedeltemperaturen i trakten är 21 °C. Den varmaste månaden är januari, då medeltemperaturen är 23 °C, och den kallaste är juli, med 18 °C. Genomsnittlig årsnederbörd är 1 794 millimeter. Den regnigaste månaden är september, med i genomsnitt 352 mm nederbörd, och den torraste är december, med 17 mm nederbörd.', """Treuhandanstalt var et stort projekt, hvor den tyske stat skulle omstrukturere erhvervslivet i det tidligere DDR. Treuhandanstalt blev etableret ved den lov (Gesetz zur Privatisierung und Reorganisation des volkseigenen Vermögens (Treuhandgesetz)), som DDR's parlament, Volkskammer, vedtog 1990 i forbindelse med aftalerne om Tysklands genforening.""", """Millenarisme er forestillinger om en fremtidig og total transformation af verden, der bæres af tilhængere af religiøse, politiske eller sociale bevægelser; oftest drejer det sig religiøse forestillinger om verdens undergang, en ny tids komme og forventningen om en snarlig udfrielse fra den normale dennesidige verden. Normalt opfattes den kommende verden som entydigt positiv, mens denne verden er ensidigt negativ hos størsteparten af de millinaristiske grupper.. Viden om de fremtidig begivenheder kan stamme fra fx åbenbaringer eller tolkning af hellige tekster, og forandringerne kan enten opfattes som positive eller negative. Millennialisme er en særlig form for millenarisme der er baseret på en idé om tusindårscykler, denne form er primært udbredt indenfor kristendommen.""", """Værnet eksperimenterede med at kurere homoseksuelle mænd med indsprøjtninger af syntetiske hormoner. SS tilbød ham i 1943 at oprette et laboratorium, hvor hans eksperimenter kunne udføres. I februar 1944 ankom han med familie til Prag, hvor laboratoriet oprettedes, og han fik titel af SS-Sturmbannfuhrer. Hans arbejde var under opsyn af Gestapos chef Heinrich Himmler, der modtog Værnets forskningsresultater.""", """Den britiske regering besluttede i 1956, at der skulle opføres en ny lufthavn ved Luqa. De finansierede en stor del af projektet, og lufthavnen åbnede den 31. marts 1958. Lufthavnsterminalen indeholdt restaurant, postkontor, et par kontorlokaler samt en udsigtsbalkon."""]
# labels = ['swe', 'dan', 'dan', 'dan', 'dan']
# # texts = ['God dag!', 'God aften!', 'Godt nytår!','God dag!','God kveld!','Godt nytt år!','God dag!','God kväll!','Gott nytt år!']
# # 'dan','dan','dan','nor','nor','nor','swe','swe','swe'

# predictor.predict(texts)
# # ['dat', 'nno', 'swe']


@app.route('/')
def index():
    welcome_msg = (
        'This is language identification API. <br>'
        'Read the docs here: <br>'
        'https://github.com/varvarvarvar/language_identification'
    )
    return welcome_msg


@app.route('/language_identification/api/v1.0/', methods=['GET'])
def forecast():

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
        {'response': prediction['response'], 'text': text})
