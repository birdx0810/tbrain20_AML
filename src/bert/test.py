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
# pylint: disable=import-error
from bert import data

# decode function
def decode(news, tokenizer, input_id, label_id):
    # get id and token mapping
    unk_token_id = tokenizer._convert_token_to_id(tokenizer.unk_token)
    pad_token_id = tokenizer._convert_token_to_id(tokenizer.pad_token)
    tokens = tokenizer.convert_ids_to_tokens(input_id)
    mapping = data.map_unk(news, tokens, input_id, unk_token_id,
                           tokenizer.all_special_tokens)

    # decode
    all_name, temp_name = [], ''
    for index, label in enumerate(label_id):
        if label == 1 and input_id[index] != pad_token_id:
            temp_name += mapping[index]
        elif label == 0 and temp_name != '':
            if len(temp_name) >= 3:
                if len(temp_name) >= 5:
                    temp_name = [temp_name[i:i+3] for i in range(0, len(temp_name), 3)]
                    all_name.extend(temp_name)
                else:
                    all_name.append(temp_name)
            temp_name = ''

    return list(set(all_name))

# predict function
def predict(news, tokenizer, model, dataloader, device):
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
    names = decode(news, tokenizer, input_id[0], prediction[0])

    return names

def test(model_path, model, news):
    # config
    if os.path.exists(f'{model_path}/config.pkl'):
        with open(f'{model_path}/config.pkl', 'rb') as f:
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
    print(f'\nload tokenizer \t time: {int(time.time())}')

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
    test_predict = predict(news, tokenizer, model, dataloader, device)
    print(f'predict result \t time: {int(time.time())}\n')

    return test_predict