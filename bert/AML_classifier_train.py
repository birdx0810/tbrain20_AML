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

# config
if os.path.exists(f'model/bert-{experiment_no}/config.pkl'):
    raise FileExistsError('Config is already exist')
else:
    args = {}

    # path
    args['experiment_no'] = experiment_no
    args['data_path'] = os.path.relpath('../data')
    args['train_file_path'] = f'{args["data_path"]}/tbrain_train_final_0610.csv'
    args['train_news_path'] = f'{args["data_path"]}/news_0703/cleaned_crawled_news'
    args['output_path'] = f'model/bert-{args["experiment_no"]}'

    if not os.path.exists(args['output_path']):
        os.makedirs(args['output_path'])

    # data
    args['test_size'] = 0.2
    args['kfold'] = 10

    # training
    args['seed'] = 7
    args['batch_size'] = 4
    args['accumulation_steps'] = 8
    args['save_steps'] = 100
    args['epochs'] = 10

    # optimizer
    args['weight_decay'] = 0.01
    args['learning_rate'] = 1e-5
    args['adam_epsilon'] = 1e-6
    args['scheduler'] = 'none'
    args['warmup_ratio'] = 0.06
    args['warmup_steps'] = 0

    # models
    args['model_name'] = 'bert-base-chinese'
    args['num_labels'] = 2
    args['max_seq_len'] = 512
    args['max_grad_norm'] = 1.0

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

# init config, model, tokenizer
config = transformers.BertConfig.from_pretrained(args['model_name'], num_labels=args['num_labels'])
model = transformers.BertForTokenClassification.from_pretrained(args['model_name'], config=config)
model = model.to(device)
tokenizer = transformers.BertTokenizer.from_pretrained(args['model_name'])

# load data
data_df = data.load_data(data_path=args['train_file_path'],
                         news_path=args['train_news_path'],
                         save_path=f'{args["data_path"]}/train.csv')
dataset = data.get_dataset(data_df, tokenizer, args)

train_data, test_data = train_test_split(dataset, test_size=args['test_size'], random_state=args['seed'])
train_dataloader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args['batch_size'],
                                               shuffle=True,
                                               collate_fn=dataset.collate_fn)
test_dataloader = torch.utils.data.DataLoader(test_data,
                                              batch_size=args['batch_size'],
                                              shuffle=False,
                                              collate_fn=dataset.collate_fn)

# optimizer
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {
        'params': [param for name, param in model.named_parameters() if not any(nd in name for nd in no_decay)],
        'weight_decay': args['weight_decay']
    },
    {
        'params': [param for name, param in model.named_parameters() if any(nd in name for nd in no_decay)],
        'weight_decay': 0.0
    }
] # ref: https://github.com/huggingface/transformers/issues/1218

optimizer = transformers.AdamW(optimizer_grouped_parameters,
                               lr=args['learning_rate'],
                               eps=args['adam_epsilon'])

# apply warm up learning rate schedule
total_steps = len(train_dataloader) // args['accumulation_steps'] * args['epochs']
warmup_steps = math.ceil(total_steps * args['warmup_ratio'])
args['warmup_steps'] = warmup_steps if args['warmup_steps'] == 0 else args['warmup_steps']

if args['scheduler'] == 'linear':
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer=optimizer,
                                                             num_warmup_steps=warmup_steps,
                                                             num_training_steps=total_steps)
elif args['scheduler'] == 'constant':
    scheduler = transformers.get_constant_schedule_with_warmup(optimizer=optimizer,
                                                             num_warmup_steps=warmup_steps)
elif args['scheduler'] == 'cosine':
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                             num_warmup_steps=warmup_steps,
                                                             num_training_steps=total_steps)
else:
    pass

# save config
if not os.path.exists(f'model/bert-{experiment_no}/config.pkl'):
    with open(f'model/bert-{experiment_no}/config.pkl', 'wb') as f:
        pickle.dump(args, f)

# create tensorboard summarywriter
writer = SummaryWriter(f'tensorboard/bert-{experiment_no}')

# evaluate function
def evaluate(model, stage, dataloader, epoch):
    loss = 0
    tqdm_desc = 'train evaluation' if stage == 'train' else 'test evaluation'
    epoch_iterator = tqdm(dataloader, total=len(dataloader), desc=tqdm_desc, position=0)
    model.eval()

    for batch in epoch_iterator:
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {
                'input_ids': batch[0], 'attention_mask': batch[1],
                'token_type_ids': batch[2], 'position_ids': batch[3],
                'labels': batch[4]
            }
            outputs = model(**inputs)
            loss += outputs[0].item()

    print(f'epoch: {epoch} evaluate {stage} loss = {loss:.10f}')

    return loss  

# train model
model.zero_grad()
model.train()

best_train_loss, best_test_loss = None, None
best_epoch = None

for epoch in range(int(args['epochs'])):
    print(f'Epoch: {epoch}')
    epoch_iterator = tqdm(train_dataloader, total=len(train_dataloader), desc='train', position=0)
    for step, train_batch in enumerate(epoch_iterator):
        train_batch = tuple(t.to(device) for t in train_batch)
        train_inputs = {
            'input_ids': train_batch[0], 'attention_mask': train_batch[1],
            'token_type_ids': train_batch[2], 'position_ids': train_batch[3],
            'labels': train_batch[4]
        }
        train_outputs = model(**train_inputs)
        loss = train_outputs[0]

        # backpropagation and update parameter
        if args['accumulation_steps'] > 1:
            loss = loss / args['accumulation_steps']
        loss.backward()

        if (step + 1) % args['accumulation_steps'] == 0:
            optimizer.step()
            if args['scheduler'] != 'none':
                scheduler.step()
            model.zero_grad()

    # evaluate and add result to tensorboard
    train_loss = evaluate(model, 'train', train_dataloader, epoch)
    test_loss = evaluate(model, 'test', test_dataloader, epoch)

    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/test', test_loss, epoch)

    if best_test_loss is None or test_loss < best_test_loss:
        best_train_loss = train_loss
        best_test_loss = test_loss
        best_epoch = epoch

    # save model
    output_dir = f'{args["output_path"]}/epoch-{epoch}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    print(f'save model to {output_dir}')

# print final result
print(f'best_epoch: {best_epoch}\n')
print(f'best_test_loss: {best_test_loss:.5f}, best_train_loss: {best_train_loss:.5f}\n')

# add hyperparameters and final result to tensorboard
writer.add_hparams({
    'test_loss': best_test_loss, 'train_loss': best_train_loss,
    'epoch': best_epoch,
    'model': args['model_name'], 'max_seq_len': args['max_seq_len'], 
    'batch_size': args['batch_size'] * args['accumulation_steps'], 'lr': args['learning_rate'],
    'weight_decay': args['weight_decay'], 'scheduler': args['scheduler']
}, metric_dict={})
writer.close()

torch.cuda.empty_cache()