# built-in modules
import os
import time

# 3rd-party modules
import numpy as np
import pickle
import torch
from tqdm import tqdm
import transformers

# self-made module
from bert import data # pylint: disable=import-error

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

    return list(set(all_name))

# predict function
def predict(tokenizer, model, dataloader, device):
    tqdm_desc = 'test predict'
    epoch_iterator = tqdm(dataloader, total=len(dataloader), desc=tqdm_desc, position=0)
    input_id, prediction = None, None

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

            if input_id is None:
                input_id = batch[0].detach().cpu().numpy()
                prediction = outputs[0].detach().cpu().numpy()
            else:
                input_id = np.append(input_id, batch[0].detach().cpu().numpy(), axis=0)
                prediction = np.append(prediction, outputs[0].detach().cpu().numpy(), axis=0)

    prediction = np.argmax(prediction, axis=2)
    names = decode(tokenizer, input_id[0], prediction[0])

    return names

def test(model, news, experiment_no, epoch):
    # config
    if os.path.exists(f'../bert/model/bert-{experiment_no}/config.pkl'):
        with open(f'../bert/model/bert-{experiment_no}/config.pkl', 'rb') as f:
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

    # init tokenizer
    tokenizer = transformers.BertTokenizer.from_pretrained(args['model_name'])
    print(f'load tokenizer \t time: {int(time.time())}')

    # load data
    dataset = data.get_dataset(news, tokenizer, args)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             collate_fn=dataset.collate_fn)
    print(f'load data \t time: {int(time.time())}')

    # move model to specific device
    model.to(device)

    # get predict result
    test_predict = predict(tokenizer, model, dataloader, device)
    print(f'predict result \t time: {int(time.time())}\n')

    return test_predict