# 3rd-party module
from flask import Flask
from flask import render_template
from flask import request
from flask import redirect
from flask import url_for
import os
import json
import transformers

# self-made module
from bert import inference
import crawl

# init flask app
app = Flask(__name__)

# BERT model
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

# define route
# predict page
@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html', AML=None)
    elif request.method == 'POST':
        # get news content
        url = request.form['url']
        origin_content = crawl.crawl_news(url)
        contents = [origin_content[i:i+512] for i in range(0, len(origin_content), 512)]

        # get AML names
        prediction = []
        for content in contents[:3]:
            prediction.extend(inference.inference(model_path, bert_model, content))

        prediction = ', '.join(prediction)

        return render_template('predict.html', url=url, content=origin_content, AML=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)