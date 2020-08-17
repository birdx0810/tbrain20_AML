# built-in modules
import os
import math
import sys

sys.path.append(f'{os.path.dirname(os.path.realpath(__file__))}/../src/')

# 3rd-party modules
import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
import transformers

from ckiptagger import data_utils, WS, POS, NER

# self-made module
import data
import scorer

# constant parameter setting
experiment_no = 1
epoch = 8
model_5fold = False

# data_path set None if use origin training data to evaluate
data_path = '../data/news.csv'
evaluate_train = True

# config
if model_5fold is True and os.path.exists(f'model_5fold/bert-{experiment_no}/config.pkl'):
    with open(f'model_5fold/bert-{experiment_no}/config.pkl', 'rb') as f:
        args = pickle.load(f)
elif model_5fold is False and os.path.exists(f'model/bert-{experiment_no}/config.pkl'):
    with open(f'model/bert-{experiment_no}/config.pkl', 'rb') as f:
        args = pickle.load(f)
else:
    raise FileNotFoundError('Config not found')

# random seed and device
device = torch.device('cpu') # pylint: disable=no-member
np.random.seed(args['seed'])
torch.manual_seed(args['seed'])

if torch.cuda.is_available():
    device = torch.device('cuda:0') # pylint: disable=no-member
    torch.cuda.manual_seed(args['seed'])
    torch.cuda.manual_seed_all(args['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# init config and tokenizer
config = transformers.BertConfig.from_pretrained(args['model_name'], num_labels=args['num_labels'])
tokenizer = transformers.BertTokenizer.from_pretrained(args['model_name'])

# load data
if data_path is None:
    data_df = data.load_data(data_path=args['train_file_path'],
                             news_path=args['train_news_path'])
else:
    data_df = data.load_data(data_path=data_path)
dataset = data.get_dataset(data_df, tokenizer, args)

if data_path is None or data_path == '../data/news.csv':
    train_data, test_data = train_test_split(dataset, test_size=args['test_size'], random_state=args['seed'])
    train_dataloader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=64,
                                                   shuffle=False,
                                                   collate_fn=dataset.collate_fn)
    test_dataloader = torch.utils.data.DataLoader(test_data,
                                                  batch_size=64,
                                                  shuffle=False,
                                                  collate_fn=dataset.collate_fn)
else:
    test_dataloader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=64,
                                                  shuffle=False,
                                                  collate_fn=dataset.collate_fn)

def decode(tokenizer, input_id, label_id):
    special_token_ids = [i for i in range(0, 106)]
    all_name, temp_name = [], ''
    for index, label in enumerate(label_id):
        if label == 1 and input_id[index] not in special_token_ids:
            temp_name += tokenizer.decode(int(input_id[index]))
        elif label == 0 and temp_name != '':
            if len(temp_name) >= 3:
                all_name.append(temp_name)
            temp_name = ''

    return set(all_name)

# predict function
def predict(tokenizer, model, stage, dataloader):
    tqdm_desc = 'train predict' if stage == 'train' else 'test predict'
    epoch_iterator = tqdm(dataloader, total=len(dataloader), desc=tqdm_desc, position=0)
    input_id, label, prediction, news = None, None, None, None

    # get label and prediction
    model.eval()
    for batch in epoch_iterator:
        batch_label = batch[-2]
        batch = tuple(t.to(device) for t in batch[:-2])

        with torch.no_grad():
            inputs = {
                'input_ids': batch[0], 'attention_mask': batch[1],
                'token_type_ids': batch[2], 'position_ids': batch[3]
            }
            outputs = model(**inputs)

            if label is None:
                input_id = batch[0].detach().cpu().numpy()
                label = batch_label
                prediction = outputs[0].detach().cpu().numpy()
            else:
                input_id = np.append(input_id, batch[0].detach().cpu().numpy(), axis=0)
                label.extend(batch_label)
                prediction = np.append(prediction, outputs[0].detach().cpu().numpy(), axis=0)

    prediction = np.argmax(prediction, axis=2)

    # decode prediction
    label = [set(name) for name in label]
    prediction = [decode(tokenizer, input_id[i], prediction[i])
                  for i in range(len(prediction))]

    return label, prediction

# load model
model = transformers.BertForTokenClassification.from_pretrained(
    f'{args["output_path"]}/epoch-{epoch}/pytorch_model.bin',
    config=config)
model.to(device)

# get predict result
if (data_path is None or data_path == '../data/news.csv') and evaluate_train:
    train_label, train_predict = predict(tokenizer, model, 'train', train_dataloader)
test_label, test_predict = predict(tokenizer, model, 'test', test_dataloader)

# calculate and print score
score = scorer.AMRScorer()
if (data_path is None or data_path == '../data/news.csv') and evaluate_train:
    train_score = score.calculate_score(train_predict, train_label)
test_score = score.calculate_score(test_predict, test_label)

print(f'Experiment {experiment_no} epoch {epoch}:')

if (data_path is None or data_path == '../data/news.csv') and evaluate_train:
    print(f'Training set:')
    print(f'Total score: {train_score:.4f} \t Average score" {train_score/len(train_label):.4f}')
print(f'Testing set:')
print(f'Total score: {test_score:.4f} \t Average score" {test_score/len(test_label):.4f}')

# save result
if (data_path is None or data_path == '../data/news.csv') and evaluate_train:
    train_df = pd.DataFrame({'No': [], 'label': [], 'predict': []})
    train_df = train_df.astype({'No': int})
    for index, (label, predict) in enumerate(zip(train_label, train_predict)):
        train_df.loc[index] = [index, list(label), list(predict)]

test_df = pd.DataFrame({'No': [], 'label': [], 'predict': []})
test_df = test_df.astype({'No': int})
for index, (label, predict) in enumerate(zip(test_label, test_predict)):
    test_df.loc[index] = [index, list(label), list(predict)]

if (data_path is None or data_path == '../data/news.csv') and evaluate_train:
    train_df.to_csv(f'{args["output_path"]}/train_predict.csv', index=False)
test_df.to_csv(f'{args["output_path"]}/test_predict.csv', index=False)

torch.cuda.empty_cache()