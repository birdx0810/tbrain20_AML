# -*- coding: UTF-8 -*-

'''
API code provided by E-SUN (garbage) and honeytoast
'''
# built-in module
import os
import json
import re
from datetime import datetime

# 3rd-party module
import flask
from flask import Flask, request, jsonify
import hashlib
import numpy as np
import pandas as pd
import transformers
from ckipnlp.container.text import TextParagraph
from ckipnlp.driver.tagger import (
    CkipTaggerWordSegmenter,
    CkipTaggerPosTagger,
    CkipTaggerNerChunker
)

# self-made module
# pylint: disable=no-member
from bert import test as bert

app = Flask(__name__)

####### PUT YOUR INFORMATION HERE #######
CAPTAIN_EMAIL = 'p76084423@gs.ncku.edu.tw'
SALT = 'ikm'

SAVE_PATH = f"./logs"
if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)

MODEL_NAME = 'BERT'
MAX_LENGTH = 512*2
DO_NER = False

############## BERT  MODEL ##############
# parameter setting
model_5fold = True
bert_experiment_no = 4
bert_epoch = 6

if model_5fold:
    model_path = f'../bert/model_5fold/bert-{bert_experiment_no}'
else:
    model_path = f'../bert/model/bert-{bert_experiment_no}'

# load config and model
bert_config = transformers.BertConfig.from_pretrained("bert-base-chinese", num_labels=2)
bert_model = transformers.BertForTokenClassification.from_pretrained(
                f'{model_path}/epoch-{bert_epoch}/pytorch_model.bin',
                config=bert_config)
print('load BERT model finished')

################## NER ##################
if DO_NER:
    ws = CkipTaggerWordSegmenter(lazy=True, disable_cuda=False)
    pos = CkipTaggerPosTagger(lazy=True, disable_cuda=False)
    ner = CkipTaggerNerChunker(lazy=True, disable_cuda=False)

    def get_ner(news):
        text = TextParagraph([news[:MAX_LENGTH]])
        ws_list = ws(text=text)
        pos_list = pos(ws=ws_list)
        ner_list = ner(ws=ws_list, pos=pos_list).to_dict()[0]
        person_list = [ner['word'] for ner in ner_list
                    if ner['ner'] == 'PERSON' and len(ner['word']) >= 2]

        return set(person_list)

#########################################

def generate_server_uuid(input_string):
    """ Create your own server_uuid
    @param input_string (str): information to be encoded as server_uuid
    @returns server_uuid (str): your unique server_uuid
    """
    s = hashlib.sha256()
    data = (input_string+SALT).encode("utf-8")
    s.update(data)
    server_uuid = s.hexdigest()

    return server_uuid

def predict(article):
    """ Predict your model result
    @param article (str): a news article
    @returns prediction (list): a list of name
    """

    ####### PUT YOUR MODEL INFERENCING CODE HERE #######
    if MODEL_NAME == "BERT":
        prediction = None
        all_news = [article[i:i+512] for i in range(0, len(article), 512)]

        for news in all_news[:3]:
            if prediction is None:
                prediction = bert.test(model_path, bert_model, news)
            else:
                prediction.extend(bert.test(model_path, bert_model, news))

        # check prediction in ner or not if DO_NER is True
        if DO_NER and prediction != []:
            person_list = get_ner(article)
            final_predict = []
            for name in prediction:
                for person in person_list:
                    if name in person:
                        final_predict.append(person)
                        continue

            prediction = list(set(final_predict))

        prediction = list(set(prediction))

    ####################################################
    prediction = _check_datatype_to_list(prediction)

    return prediction

def _check_datatype_to_list(prediction):
    """ Check if your prediction is in list type or not. 
        And then convert your prediction to list type or raise error.
        
    @param prediction (list / numpy array / pandas DataFrame): your prediction
    @returns prediction (list): your prediction in list type
    """
    if isinstance(prediction, np.ndarray):
        _check_datatype_to_list(prediction.tolist())
    elif isinstance(prediction, pd.core.frame.DataFrame):
        _check_datatype_to_list(prediction.values)
    elif isinstance(prediction, list):
        return prediction
    raise ValueError('Prediction is not in list type.')

@app.route('/healthcheck', methods=['POST'])
def healthcheck():
    """ API for health check """
    try:
        # get request data
        data = request.get_json(force=True)
        print(f'\n{data}\n')

        # generate server uuid
        server_uuid = generate_server_uuid(CAPTAIN_EMAIL)

        # get response time
        server_timestamp = int(datetime.now().timestamp())

        return jsonify({'esun_uuid': data['esun_uuid'],
                        'server_uuid': server_uuid,
                        'captain_email': CAPTAIN_EMAIL,
                        'server_timestamp': server_timestamp})
    except Exception as err:
        print(err)

@app.route('/inference', methods=['POST'])
def inference():
    """ API that return your model predictions when E.SUN calls this API """
    # get request data
    data = request.get_json(force=True)

    # generate server uuid
    server_uuid = generate_server_uuid(CAPTAIN_EMAIL)

    # get request time
    esun_timestamp = datetime.fromtimestamp(data['esun_timestamp'])
    start_timestamp = datetime.now()

    # remove unnecessary character
    news = data['news']
    news = re.sub(r'\s+', '', news)

    # get answer
    try:
        answer = predict(news)
    except:
        raise ValueError('Model error.')
    
    # get response time
    end_timestamp = datetime.now()

    # write log
    with open(f'{SAVE_PATH}/{data["esun_uuid"]}.log', 'w') as f:
        f.write(f'ESUN TIME: {datetime.strftime(esun_timestamp, "%Y-%d-%m %H:%M:%S.%f")}\n')
        f.write(f'STR TIME: {datetime.strftime(start_timestamp, "%Y-%d-%m %H:%M:%S.%f")}\n')
        f.write(f'END TIME: {datetime.strftime(end_timestamp, "%Y-%d-%m %H:%M:%S.%f")}\n')
        f.write(f'{news}\n')
        if answer != []:
            f.write(f'{",".join(answer)}')
        else:
            f.write('\n')

    # print data
    data['answer'] = answer
    print(f'\n{data}\n')

    return jsonify({'esun_uuid': data['esun_uuid'],
                    'server_uuid': server_uuid,
                    'answer': answer,
                    'server_timestamp': int(end_timestamp.timestamp())})

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8080, debug=True, threaded=True)