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
    def __init__(self, input_size, output_size):
        super(TemporalDecay, self).__init__()
        self.build(input_size, output_size)

    def build(self, input_size, output_size):
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, delta):
        gamma = F.relu(F.linear(delta, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma

class FeatureRegression(nn.Module):
    def __init__(self, input_size):
        super(FeatureRegression, self).__init__()
        self.build(input_size)

    def build(self, input_size):
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))

        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)
        self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        z_h = F.linear(x, self.W * Variable(self.m), self.b)
        return z_h


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.build()

    def build(self):
        self.rnn_cell = nn.GRUCell(36 * 3, RNN_HID_SIZE)

        self.history_regression = nn.Linear(RNN_HID_SIZE, 1)
        self.feature_regression = FeatureRegression(36)

        self.temp_decay = TemporalDecay(input_size = 36, output_size = 36)
        self.temp_decay_h = TemporalDecay(input_size = 36, output_size = 64)

        self.combine = nn.Linear(36 * 2, 36)

        self.out = nn.Linear(RNN_HID_SIZE, 1)

    def forward(self, data, dir_, alpha = 1.0):
        values = data[dir_]['values']
        masks = data[dir_]['masks']
        deltas = data[dir_]['deltas']

        label = data['label'].view(-1, 1)
        is_train = data['is_train'].view(-1, 1)

        h = Variable(torch.zeros((values.size()[0], RNN_HID_SIZE)))
        h_sum = Variable(torch.zeros((values.size()[0], RNN_HID_SIZE)))

        if torch.cuda.is_available():
            h = h.cuda()
            h_sum = h_sum.cuda()

        impute_loss = 0.0

        impute_values = []

        for t in range(values.size()[1]):
            value = values[:, t]
            mask = masks[:, t]
            delta = deltas[:, t]

            value_x = self.history_regression(h)
            impute_loss += torch.sum(torch.abs(value_x - value) * mask) / (torch.sum(mask) + EPS)

            value_c = mask * value + (1 - mask) * value_x

            gamma = self.temp_decay(delta)

            beta = F.sigmoid(self.combine(torch.cat([gamma, mask], dim = 1)))

            value_z = self.feature_regression(value_c)
            impute_loss += torch.sum(torch.abs(value_z - value) * mask) / (torch.sum(mask) + EPS)

            value_h = value_z * beta + value_x * (1 - beta)
            impute_loss += torch.sum(torch.abs(value_h - value) * mask) / (torch.sum(mask) + EPS)

            value_c = mask * value + (1 - mask) * value_h

            impute_values.append(value_c)

            inputs = torch.cat([value_c, mask, gamma], dim = 1)

            h = self.rnn_cell(inputs, h)
            h_sum = h_sum + h

        impute_loss = impute_loss / values.size()[1]

        impute_values = torch.cat(impute_values, dim = 1)

        h_sum = h_sum / values.size()[1]
        pred = self.out(h_sum)

        class_loss = utils.binary_cross_entropy_with_logits(pred, label, reduce = False)
        class_loss = torch.sum(class_loss * is_train) / (torch.sum(is_train) + EPS)

        loss = class_loss + impute_loss * alpha

        pred = F.sigmoid(pred)

        return {'loss': loss, 'impute_values': impute_values, 'pred': pred}

    def run_on_batch(self, data, optimizer, epoch = 0):
        ret = self(data, dir_ = 'forward', alpha = 1.0)

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret
