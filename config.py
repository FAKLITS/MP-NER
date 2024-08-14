import os
import torch
data_dir = os.getcwd() + '\\data\\clue\\'
train_dir = data_dir + 'train.npz'
test_dir = data_dir + 'test.npz'
files = ['train', 'test']
roberta_model = 'pretrained_bert_models/chinese_roberta_wwm_large_ext/'
max_vocab_size = 1000000
dev_split_size = 0.25
full_fine_tuning = True
learning_rate = 3e-5
weight_decay = 0.01
clip_grad = 5
batch_size = 12
epoch_num = 50
min_epoch_num = 25
patience = 0.0002
patience_num = 10
gpu = ' '
if gpu != '':
    device = torch.device(f"cuda:{gpu}")
else:
    device = torch.device("cpu")
labels = ['CONT', 'EDU', 'LOC', 'NAME', 'ORG', 'PRO', 'RACE', 'TITLE']
label2id = {
    "O": 0,
    "B-CONT": 1,
    "B-EDU": 2,
    "B-LOC": 3,
    'B-NAME': 4,
    'B-ORG': 5,
    'B-PRO': 6,
    'B-RACE': 7,
    'B-TITLE': 8,
    "I-CONT": 9,
    "I-EDU": 10,
    "I-LOC": 11,
    'I-NAME': 12,
    'I-ORG': 13,
    'I-PRO': 14,
    'I-RACE': 15,
    'I-TITLE': 16,
}
id2label = {_id: _label for _label, _id in list(label2id.items())}
