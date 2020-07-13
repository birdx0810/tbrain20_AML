# built-in modules
import os
import math

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

# self-made module
import data

# constant parameter setting
experiment_no = 1
epoch = 8
evaluate_train = False

# config
if os.path.exists(f'model/bert-{experiment_no}/config.pkl'):
    with open(f'model/bert-{experiment_no}/config.pkl', 'rb') as f:
        args = pickle.load(f)
else:
    raise FileNotFoundError('Config not found')

# random seed and device
device = torch.device('cpu')
np.random.seed(args['seed'])
torch.manual_seed(args['seed'])

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.manual_seed(args['seed'])
    torch.cuda.manual_seed_all(args['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# init config and tokenizer
config = transformers.BertConfig.from_pretrained(args['model_name'], num_labels=args['num_labels'])
tokenizer = transformers.BertTokenizer.from_pretrained(args['model_name'])

# load data
data_df = data.load_data(data_path=args['train_file_path'],
                         news_path=args['train_news_path'],
                         save_path=f'{args["data_path"]}/train.csv')
dataset = data.get_dataset(data_df, tokenizer, args)

train_data, test_data = train_test_split(dataset, test_size=args['test_size'], random_state=args['seed'])
train_dataloader = torch.utils.data.DataLoader(train_data,
                                            #    batch_size=args['batch_size'],
                                               batch_size=64,
                                               shuffle=False,
                                               collate_fn=dataset.collate_fn)
test_dataloader = torch.utils.data.DataLoader(test_data,
                                            #   batch_size=args['batch_size'],
                                              batch_size=64,
                                              shuffle=False,
                                              collate_fn=dataset.collate_fn)

# decode function
def decode(tokenizer, input_id, label_id):
    all_name, temp_name = [], ''
    for index, label in enumerate(label_id):
        if label == 1:
            temp_name += tokenizer.decode(int(input_id[index]))
        elif label == 0:
            if temp_name != '':
                all_name.append(temp_name)
                temp_name = ''

    return all_name

# predict function
def predict(tokenizer, model, stage, dataloader):
    tqdm_desc = 'train predict' if stage == 'train' else 'test predict'
    epoch_iterator = tqdm(dataloader, total=len(dataloader), desc=tqdm_desc, position=0)
    input_id, label, prediction = None, None, None

    # get label and prediction
    model.eval()
    for batch in epoch_iterator:
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {
                'input_ids': batch[0], 'attention_mask': batch[1],
                'token_type_ids': batch[2], 'position_ids': batch[3]
            }
            outputs = model(**inputs)

            if label is None:
                input_id = batch[0].detach().cpu().numpy()
                label = batch[4].detach().cpu().numpy()
                prediction = outputs[0].detach().cpu().numpy()
            else:
                input_id = np.append(input_id, batch[0].detach().cpu().numpy(), axis=0)
                label = np.append(label, batch[4].detach().cpu().numpy(), axis=0)
                prediction = np.append(prediction, outputs[0].detach().cpu().numpy(), axis=0)

    prediction = np.argmax(prediction, axis=2)

    # decode prediction
    all_label = [decode(tokenizer, input_id[i], label[i])
                 for i in range(len(label))]
    all_prediction = [decode(tokenizer, input_id[i], prediction[i])
                      for i in range(len(prediction))]

    return all_label, all_prediction

# load model
model = transformers.BertForTokenClassification.from_pretrained(
    f'model/bert-{experiment_no}/epoch-{epoch}/pytorch_model.bin',
    config=config)
model.to(device)

# get predict result
if evaluate_train:
    train_label, train_predict = predict(tokenizer, model, 'train', train_dataloader)
test_label, test_predict = predict(tokenizer, model, 'test', test_dataloader)

# print result
if evaluate_train:
    print('train result')
    for index, (label, predict) in enumerate(zip(train_label, train_predict)):
        print(f"{'-'*50}")
        print(f"{index}|label:  \t{label}")
        print(f"{index}|predict:\t{predict}")

print('test result')
for index, (label, predict) in enumerate(zip(test_label, test_predict)):
    print(f"{'-'*50}")
    print(f"{index}|label:  \t{set(label)}")
    print(f"{index}|predict:\t{set(predict)}")