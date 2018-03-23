import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import time
import utils
import models
import argparse
import data_loader
import pandas as pd
import ujson as json

from sklearn import metrics

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type = int, default = 100)
parser.add_argument('--batch_size', type = int, default = 64)
parser.add_argument('--model', type = str)
parser.add_argument('--train_file', type = str, default = './data/train')
parser.add_argument('--val_file', type = str, default = './data/test')
args = parser.parse_args()

EPS = 1e-5

def train(model):
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)

    # we only use X_{test}, y_{test} is not visible to the model
    train_iter = data_loader.get_loader([args.train_file, args.val_file], batch_size = args.batch_size, shuffle = True)
    val_iter = data_loader.get_loader([args.val_file], batch_size = args.batch_size, shuffle = False)

    for epoch in xrange(args.epochs):
        model.train()

        running_loss = 0.0

        for idx, data in enumerate(train_iter):
            data = utils.to_var(data)
            ret = model.run_on_batch(data, optimizer, epoch = epoch)

            running_loss += ret['loss'].data[0]

            print '\r Progress epoch {}, {:.2f}%, average loss {}'.format(epoch, (idx + 1) * 100.0 / len(train_iter), running_loss / (idx + 1.0)),

        evaluate(model, val_iter)

def evaluate(model, val_iter):
    model.eval()

    pred = []
    label = []

    running_loss = 0.0

    for idx, data in enumerate(val_iter):
        data = utils.to_var(data)
        ret = model.run_on_batch(data, optimizer = None)

        running_loss += ret['loss'].data[0]

        pred += ret['pred'].data.tolist()
        label += data['label'].data.view(-1, 1).tolist()

    pred = np.asarray(pred)
    label = np.asarray(label).astype('int32')

    print 'Eval loss {}, AUC {}'.format(running_loss / len(val_iter), metrics.roc_auc_score(label, pred))

def run():
    model = getattr(models, args.model).Model()

    if torch.cuda.is_available():
        model = model.cuda()

    train(model)

if __name__ == '__main__':
    run()
