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

import RITS
from sklearn import metrics

RNN_HID_SIZE = 64


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.build()

    def build(self):
        self.rits_f = RITS.Model()
        self.rits_b = RITS.Model()

    def forward(self, data):
        ret_f = self.rits_f(data, dir_ = 'forward')
        ret_b = self.reverse(self.rits_b(data, dir_ = 'backward'))

        ret = self.merge_ret(ret_f, ret_b)

        return ret

    def merge_ret(self, ret_f, ret_b):
        loss_f = ret_f['loss']
        loss_b = ret_b['loss']
        loss_c = self.get_consistency_loss(ret_f['impute_values'], ret_b['impute_values'])

        loss = loss_f + loss_b + loss_c

        impute_values = (ret_f['impute_values'] + ret_b['impute_values']) / 2
        pred = (ret_f['pred'] + ret_b['pred']) / 2

        ret_f['loss'] = loss
        ret_f['pred'] = pred
        ret_f['impute_values'] = impute_values

        return ret_f

    def get_consistency_loss(self, pred_f, pred_b):
        loss = torch.pow(pred_f - pred_b, 2).mean()
        return loss

    def reverse(self, ret):
        def reverse_tensor(tensor_):
            if tensor_.dim() <= 1:
                return tensor_
            indices = range(tensor_.size()[1])[::-1]
            indices = torch.LongTensor(indices)

            if torch.cuda.is_available():
                indices = Variable(indices.cuda(), requires_grad = False)

            return tensor_.index_select(1, indices)

        for key in ret:
            ret[key] = reverse_tensor(ret[key])

        return ret

    def run_on_batch(self, data, optimizer, epoch = 0):
        ret = self(data)

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret

