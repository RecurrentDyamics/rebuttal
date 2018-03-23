import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.nn.parameter import Parameter

import math
import utils
import argparse
import data_loader

from ipdb import set_trace
from sklearn import metrics

EPS = 1e-5
RNN_HID_SIZE = 64

class TemporalDecay(nn.Module):
    def __init__(self, input_size, output_size, diagonal = False):
        super(TemporalDecay, self).__init__()

        self.diagonal = diagonal

        self.build(input_size, output_size)

    def build(self, input_size, output_size):
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))
        self.reset_parameters()

        if self.diagonal:
            self.e = Variable(torch.eye(output_size), requires_grad = False)
            if torch.cuda.is_available():
                self.e = self.e.cuda()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, delta):
        if self.diagonal:
            gamma = F.relu(F.linear(delta, self.W * self.e, self.b))
        else:
            gamma = F.relu(F.linear(delta, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.build()

    def build(self):
        self.temp_decay_x = TemporalDecay(input_size = 36, output_size = 36, diagonal = True)
        self.temp_decay_h = TemporalDecay(input_size = 36, output_size = RNN_HID_SIZE, diagonal = False)

        self.gru_cell = nn.GRUCell(36 * 2, 64)
        self.gru_drop = nn.Dropout(0.3)

        self.out_drop = nn.Dropout(0.5)
        self.out = nn.Linear(64, 1)

    def forward(self, data, dir_, alpha):
        values = data[dir_]['values']
        masks = data[dir_]['masks']
        deltas = data[dir_]['deltas']
        lasts = data[dir_]['lasts']

        label = data['label'].view(-1, 1)
        is_train = data['is_train'].view(-1, 1)

        impute_values = []

        h = Variable(torch.zeros((values.size()[0], RNN_HID_SIZE)))

        if torch.cuda.is_available():
            h = h.cuda()

        for t in range(values.size()[1]):
            value = values[:, t]
            mask = masks[:, t]
            delta = deltas[:, t]
            last = lasts[:, t]

            gamma_x = self.temp_decay_x(delta)

            value_h = mask * value + (1 - mask) * gamma_x * last
            impute_values.append(value_h.unsqueeze(dim = 1))

            input_ = torch.cat([value_h, mask], dim = 1)

            h = h * self.temp_decay_h(delta)
            h = self.gru_drop(h)

            h = self.gru_cell(input_, h)

        impute_values = torch.cat(impute_values, dim = 1)

        pred = self.out(h)

        class_loss = utils.binary_cross_entropy_with_logits(pred, label, reduce = False)
        class_loss = torch.sum(class_loss * is_train) / (torch.sum(is_train) + EPS)

        pred = F.sigmoid(pred)

        return {'loss': class_loss, 'impute_values': impute_values, 'pred': pred}

    def run_on_batch(self, data, optimizer, epoch = 0):
        ret = self(data, dir_ = 'forward', alpha = 0)

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret
