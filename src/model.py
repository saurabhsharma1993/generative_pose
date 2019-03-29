#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch
from torch.autograd import Variable


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal(m.weight)

# basic residual block
class ResidualBlock(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super(ResidualBlock, self).__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out

class CVAE_Linear(nn.Module):
    def __init__(self, size, latent_size = 512, num_samples=10, alpha=1, cvae_num_stack=1):
        super(CVAE_Linear, self).__init__()

        self.size = size
        self.latent_size = latent_size
        self.cvae_num_stack = cvae_num_stack

        self.inp_post = nn.Linear(16*2 + 16*3, size)
        self.out_post = nn.Linear(size, self.latent_size*2)

        # use this for customizable num of stacks
        enc_post = []
        for i in range(cvae_num_stack):
            enc_post.append(ResidualBlock(size))
        self.enc_post = nn.Sequential(*enc_post)

        self.fc_dec = nn.Linear(16 * 2, self.size - self.latent_size)

        # use this for customizable num of stacks
        dec = []
        for i in range(cvae_num_stack):
            dec.append(ResidualBlock(size))
        self.dec = nn.Sequential(*dec)

        self.fc_out = nn.Linear(size, 16*3)

        # additional layers
        self.batch_norm1 = nn.BatchNorm1d(self.size)
        self.batch_norm2 = nn.BatchNorm1d(self.latent_size*2)
        self.batch_norm3 = nn.BatchNorm1d(self.size - self.latent_size)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()

        self.num_samples = num_samples
        self.alpha = alpha
        self.type = type

    def encode(self, x, y):

        post_inp = torch.cat([x, y], dim=1)
        post_inp = self.inp_post(post_inp)
        post_inp = self.batch_norm1(post_inp)
        post_inp = self.relu(post_inp)
        post_inp = self.dropout(post_inp)

        post = self.enc_post(post_inp)
        post = self.out_post(post)
        post = self.batch_norm2(post)
        post = self.relu(post)
        post = self.dropout(post)

        post_mean, post_logvar = torch.split(post,self.latent_size,dim=1)

        return post_mean, post_logvar

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def decode(self, z, x):

        x = self.fc_dec(x)
        x = self.batch_norm3(x)
        x = self.relu(x)
        x = self.dropout(x)
        dec_inp = torch.cat([z, x], dim=1)
        out = self.dec(dec_inp)
        out = self.fc_out(out)

        return out

    def forward(self, x, y):

        self.batch_size = x.shape[0]
        post_mu, post_logvar = self.encode(x, y)

        out = torch.zeros(self.num_samples,self.batch_size,16*3).cuda()
        out_gsnn = torch.zeros(self.num_samples, self.batch_size, 16 * 3).cuda()

        for i in range(self.num_samples):
            z = self.reparameterize(post_mu, post_logvar)
            out[i] = torch.unsqueeze(self.decode(z, x),0)

        if (self.alpha != 1):
            for i in range(self.num_samples):
                z = torch.randn(self.batch_size, self.latent_size).cuda()
                out_gsnn[i] = self.decode(z, x)

        return torch.mean(out, dim=0), torch.mean(out_gsnn, dim=0), post_mu, post_logvar