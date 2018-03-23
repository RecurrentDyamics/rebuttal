import os
import time

import ujson as json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class MySet(Dataset):
    def __init__(self, input_files):
        super(MySet, self).__init__()
        self.content = []

        for input_file in input_files:
            self.content += open(input_file).readlines()

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        return self.content[idx]

def collate_fn(recs):
    recs = map(lambda x: json.loads(x), recs)

    forward = map(lambda x: x['forward'], recs)
    backward = map(lambda x: x['backward'], recs)

    def to_tensor_dict(recs):
        values = torch.FloatTensor(map(lambda x: x['values'], recs))
        masks = torch.FloatTensor(map(lambda x: x['masks'], recs))
        deltas = torch.FloatTensor(map(lambda x: x['deltas'], recs))
        lasts = torch.FloatTensor(map(lambda x: x['lasts'], recs))

        return {'values': values, 'masks': masks, 'deltas': deltas, 'lasts': lasts}

    ret_dict = {'forward': to_tensor_dict(forward), 'backward': to_tensor_dict(backward)}

    ret_dict['label'] = torch.FloatTensor(map(lambda x: x['label'], recs))
    ret_dict['is_train'] = torch.FloatTensor(map(lambda x: x['is_train'], recs))

    return ret_dict

def get_loader(input_files, batch_size = 64, shuffle = True):
    data_set = MySet(input_files)
    data_iter = DataLoader(dataset = data_set, \
                              batch_size = batch_size, \
                              num_workers = 8, \
                              shuffle = shuffle, \
                              pin_memory = True, \
                              collate_fn = collate_fn
    )

    return data_iter

if __name__ == '__main__':
    data_iter = get_loader('./data/test')
    for idx, item in enumerate(data_iter):
        print item['forward']
