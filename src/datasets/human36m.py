#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import

import os
import torch
from torch.utils.data import Dataset
import numpy as np
import src.utils as utils


class Human36M(Dataset):
    def __init__(self, actions, data_path, is_train=True, procrustes=False):

        self.actions = actions
        self.data_path = data_path
        self.procrustes = procrustes
        self.is_train = is_train

        self.train_inp, self.train_out, self.test_inp, self.test_out, self.pred_ord = [], [], [], [], []
        self.train_meta, self.test_meta = [], []

        train_2d_file = 'train_2d_ft_norm.pth.pt'
        test_2d_file = 'test_2d_ft_norm.pth.pt'
        test_ord_file = 'test_2d_ft_ord.pth.pt'

        if self.is_train:
            # load train data
            self.train_3d = torch.load(os.path.join(data_path, 'train_3d.pth.pt'))
            self.train_2d = torch.load(os.path.join(data_path, train_2d_file))
            for k2d in self.train_2d.keys():
                (sub, act, fname) = k2d
                k3d = k2d
                k3d = (sub, act, fname[:-3]) if fname.endswith('-sh') else k3d
                num_f, _ = self.train_2d[k2d].shape
                assert self.train_3d[k3d].shape[0] == self.train_2d[k2d].shape[0], '(training) 3d & 2d shape not matched'
                for i in range(num_f):
                    self.train_inp.append(self.train_2d[k2d][i])
                    self.train_out.append(self.train_3d[k3d][i])

            if(self.procrustes):
                self.test_3d = torch.load(os.path.join(data_path, 'test_3d.pth.pt'))
                self.test_2d = torch.load(os.path.join(data_path, test_2d_file))
                for k2d in self.test_2d.keys():
                    (sub, act, fname) = k2d
                    if(not sub==9):
                        continue # subject 9 is also used in training for this protocol
                    k3d = k2d
                    k3d = (sub, act, fname[:-3]) if fname.endswith('-sh') else k3d
                    num_f, _ = self.test_2d[k2d].shape
                    assert self.test_3d[k3d].shape[0] == self.test_3d[k2d].shape[0], '(training) 3d & 2d shape not matched'
                    for i in range(num_f):
                        self.train_inp.append(self.test_2d[k2d][i])
                        self.train_out.append(self.test_3d[k3d][i])
        else:
            # load test data
            self.test_3d = torch.load(os.path.join(data_path, 'test_3d.pth.pt'))
            self.test_2d = torch.load(os.path.join(data_path, test_2d_file))
            self.test_ord = torch.load(os.path.join(data_path, test_ord_file))
            for k2d in self.test_2d.keys():
                (sub, act, fname) = k2d
                if act not in self.actions:
                    continue
                if (self.procrustes and sub==9):
                    continue # this prot only evaluates on subject 11
                k3d = k2d
                k3d = (sub, act, fname[:-3]) if fname.endswith('-sh') else k3d

                if(k2d not in self.test_ord['num_frames']):
                    continue
                num_f = self.test_ord['num_frames'][k2d]
                assert self.test_2d[k2d].shape[0] == self.test_3d[k3d].shape[0], '(test) 3d & 2d shape not matched'
                for i in range(num_f):
                    self.test_inp.append(self.test_2d[k2d][i])
                    self.test_out.append(self.test_3d[k3d][i])
                    self.pred_ord.append(self.test_ord['joint_ordinal'][k2d][i])

    def __getitem__(self, index):
        SH_TO_GT_PERM = np.array([utils.SH_NAMES.index(h) for h in utils.H36M_NAMES if h != '' and h in utils.SH_NAMES])
        GT_TO_SH_PERM = np.array([utils.H36M_NAMES.index(h) for h in utils.SH_NAMES if h != '' and h in utils.H36M_NAMES])

        if self.is_train:
            inputs = torch.from_numpy(self.train_inp[index]).float()
            outputs = torch.from_numpy(self.train_out[index]).float()
            ordinals = torch.from_numpy(np.zeros((16,16)))
        else:
            inputs = torch.from_numpy(self.test_inp[index]).float()
            outputs = torch.from_numpy(self.test_out[index]).float()
            ordinals = torch.from_numpy(self.pred_ord[index]).float()

        return inputs, outputs, ordinals

    def __len__(self):
        if self.is_train:
            return len(self.train_inp)
        else:
            return len(self.test_inp)
