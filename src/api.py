# -*- coding: UTF-8 -*-

'''
API code provided by E-SUN (garbage) and honeytoast
'''
# built-in module
import os
import time
import json
import re

# 3rd-party module
import flask
from flask import Flask, request, jsonify
import hashlib
import numpy as np
import pandas as pd
import transformers

# self-made module
# pylint: disable=no-member
from bert import test as bert

app = Flask(__name__)

####### PUT YOUR INFORMATION HERE #######
CAPTAIN_EMAIL = 'p76084423@gs.ncku.edu.tw'
SALT = 'ikm'

INFERENCE_COUNT = 0
SAVE_PATH = f"./logs"
if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)

MODEL_NAME = 'BERT'

############## BERT  MODEL ##############
# parameter setting
bert_experiment_no = 1
bert_epoch = 8

# load config and model
bert_config = transformers.BertConfig.from_pretrained("bert-base-chinese", num_labels=2)
bert_model = transformers.BertForTokenClassification.from_pretrained(
    f'../bert/model/bert-{bert_experiment_no}/epoch-{bert_epoch}/pytorch_model.bin',
    config=bert_config)
print('load BERT model finished')

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
        prediction = bert.test(bert_model, article, 
                               bert_experiment_no, bert_epoch)

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
        server_timestamp = int(time.time())

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
    esun_timestamp = data['esun_timestamp']
    start_timestamp = int(time.time())

    # remove unnecessary character
    news = data['news']
    news = re.sub(r'\s+', '', news)

    # get answer
    try:
        answer = predict(news)
    except:
        raise ValueError('Model error.')        
    
    # get response time
    end_timestamp = int(time.time())

    global INFERENCE_COUNT

    # write log
    with open(f'{SAVE_PATH}/{data["esun_uuid"]}.log', 'w') as f:
        f.write(f'ESUN TIME: {time.ctime(esun_timestamp)}\n')
        f.write(f'STR TIME: {time.ctime(start_timestamp)}\n')
        f.write(f'END TIME: {time.ctime(end_timestamp)}\n')
        f.write(f'{news}\n')
        if answer != []:
            f.write(f'{",".join(answer)}')
        else:
            f.write('\n')

    return jsonify({'esun_uuid': data['esun_uuid'],
                    'server_uuid': server_uuid,
                    'answer': answer,
                    'server_timestamp': end_timestamp})

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8080, debug=False)