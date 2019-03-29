#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division

import os
import sys
from pprint import pprint
import numpy as np

import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F

from opt import Options
import src.utils as utils
import src.log as log

from src.model import CVAE_Linear, weight_init
from src.datasets.human36m import Human36M

def loss_function(y, y_gsnn, x, mu, logvar):

    L2_cvae = option.alpha * F.mse_loss(y, x)
    L2_gsnn = (1 - option.alpha) * F.mse_loss(y_gsnn, x)
    L2 = L2_cvae + L2_gsnn
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return L2, L2_cvae, L2_gsnn, KLD

def train_multiposenet(train_loader, model, criterion, optimizer, lr_init=None, lr_now=None, glob_step=None, lr_decay=None, gamma=None, max_norm=True):

    model.train()
    l2_loss, cvae_loss, gsnn_loss, kl_loss = utils.AverageMeter(), utils.AverageMeter(), utils.AverageMeter(), utils.AverageMeter()

    for i, (inps, tars, _) in enumerate(train_loader):

        glob_step += 1
        if glob_step % lr_decay == 0 or glob_step == 1:
            lr_now = utils.lr_decay(optimizer, glob_step, lr_init, lr_decay, gamma)

        # forward pass
        inputs = Variable(inps.cuda())
        targets = Variable(tars.cuda())
        out_cvae, out_gsnn, post_mu, post_logvar = model(inputs, targets)

        # backward pass
        optimizer.zero_grad()
        loss_l2, loss_cvae, loss_gsnn, loss_kl = loss_function(out_cvae, out_gsnn, targets, post_mu, post_logvar)
        loss_l2 = loss_l2 * option.weight_l2
        loss_cvae = loss_cvae * option.weight_l2
        loss_gsnn = loss_gsnn * option.weight_l2
        loss_kl = loss_kl * option.weight_kl

        l2_loss.update(loss_l2.item(), inputs.size(0))
        cvae_loss.update(loss_cvae.item(), inputs.size(0))
        gsnn_loss.update(loss_gsnn.item(), inputs.size(0))
        kl_loss.update(loss_kl.item(), inputs.size(0))

        loss = loss_kl + loss_l2
        loss.backward()

        if max_norm:
            nn.utils.clip_grad_norm(model.parameters(), max_norm=1)
        optimizer.step()

        # update summary
        if (i % 100 == 0):

            print('({batch}/{size}) | loss l2: {loss_l2:.4f} | loss cvae: {loss_cvae:.4f} | loss gsnn: {loss_gsnn:.4f} | loss kl: {loss_kl:.4f}' \
                    .format(batch=i + 1,
                            size=len(train_loader),
                            loss_l2=l2_loss.avg,
                            loss_cvae=cvae_loss.avg,
                            loss_gsnn=gsnn_loss.avg,
                            loss_kl=kl_loss.avg))

            sys.stdout.flush()

    return glob_step, lr_now, l2_loss.avg

def test_multiposenet(test_loader, model, criterion, stat_3d, stat_2d, procrustes=False):

    model.eval()
    l2_loss = utils.AverageMeter()

    # global error trackers
    all_dist, all_dist_samples, all_dist_ordsamp_weighted, all_dist_ordsamp_weighted_pred = [], [], {}, {}
    temp_gt = np.linspace(0.1, 1, num=10) # range of temperatures for softmax in OrdinalScore
    temp_pred = np.linspace(0.1, 1, num=10)
    for ind, t in enumerate(temp_gt):
        all_dist_ordsamp_weighted[ind] = []
        all_dist_ordsamp_weighted_pred[ind] = []

    for i, (inps, tars, ordinals) in enumerate(test_loader):

        if (not i % 20 == 0): # for quick validation during training
            continue

        inputs = Variable(inps.cuda())
        targets = Variable(tars.cuda(async=True))

        batch_size = inputs.shape[0]
        z_samples, out_samples = [], []
        num_samples = option.numSamples

        # generate sample set
        for j in range(num_samples):

            z = torch.randn(batch_size, option.latent_size).cuda()
            z_samples.append(z)

            out = model.decode(z, inputs)
            out_samples.append(out)

            loss_l2, _, _, loss_kl = loss_function(out, out, targets,
                                                   torch.zeros((option.test_batch, option.latent_size)).cuda(),
                                                   torch.zeros((option.test_batch, option.latent_size)).cuda())
            loss_l2 = loss_l2 * option.weight_l2
            loss = loss_kl + loss_l2

            l2_loss.update(loss_l2.item(), inputs.size(0))

        out_samp = torch.cat([torch.unsqueeze(out_sample, dim=0) for out_sample in out_samples])
        out_mean = torch.mean(out_samp, dim=0)
        tars = targets

        # unnormalise everything and slice along used dimensions
        inps_unnorm = utils.unNormalizeData(inps.data.cpu().numpy(), stat_2d['mean'], stat_2d['std'], stat_2d['dim_use'])
        targets_unnorm = utils.unNormalizeData(tars.data.cpu().numpy(), stat_3d['mean'], stat_3d['std'], stat_3d['dim_use'])
        outputs_unnorm = utils.unNormalizeData(out_mean.data.cpu().numpy(), stat_3d['mean'], stat_3d['std'], stat_3d['dim_use'])
        outputs_samples_unnorm = np.vstack([utils.unNormalizeData(out_sample.data.cpu().numpy(), stat_3d['mean'], stat_3d['std'], stat_3d['dim_use'])[None] for out_sample in out_samples])

        dim_use = np.hstack((np.arange(3), stat_3d['dim_use']))
        outputs_samples_use = outputs_samples_unnorm[:, :, dim_use]
        outputs_use = outputs_unnorm[:, dim_use]
        targets_use = targets_unnorm[:, dim_use]

        # procrustes alignment
        if (procrustes):
            procrustes_outputs_use = np.zeros(outputs_use.shape)
            procrustes_outputs_samples_use = np.zeros(outputs_samples_use.shape)
            for ba in range(inps.size(0)):
                gt = targets_use[ba].reshape(-1, 3)
                out = outputs_use[ba].reshape(-1, 3)
                _, Z, T, b, c = utils.get_transformation(gt, out, True)
                out = (b * out.dot(T)) + c
                procrustes_outputs_use[ba, :] = out.reshape(1, 51)
                for k in range(num_samples):
                    out = outputs_samples_use[k, ba].reshape(-1, 3)
                    _, Z, T, b, c = utils.get_transformation(gt, out, True)
                    out = (b * out.dot(T)) + c
                    procrustes_outputs_samples_use[k, ba, :] = out.reshape(1, 51)

            outputs_use = procrustes_outputs_use
            outputs_samples_use = procrustes_outputs_samples_use

        # OrdinalScore
        GT_TO_SH_PERM = np.array([utils.H36M_NAMES.index(h) for h in utils.SH_NAMES if h != '' and h in utils.H36M_NAMES])
        gt_ord = utils.compute_ordinals(targets_unnorm.reshape(-1, 32, 3)[:, GT_TO_SH_PERM, 2], 1) # compute ground truth ordinal relations
        pred_ord = ordinals.data.cpu().numpy() # estimated ordinal relations from OrdinalNet
        samp_ord = utils.compute_ordinals(outputs_samples_unnorm.reshape(-1, batch_size, 32, 3)[:, :, GT_TO_SH_PERM, 2], 1) # compute ordinal relations for generated samples
        score_ord_gt = utils.compare(samp_ord, gt_ord) # OrdinalScore using GT ordinals
        score_ord_pred = utils.compare(samp_ord, utils.postproc(pred_ord)) # OrdinalScore using PRED ordinals

        score_ord_softmax_gt = torch.zeros((temp_gt.shape[0], num_samples, batch_size))
        weighted_preds_gt = np.zeros((temp_gt.shape[0], batch_size, 51))
        score_ord_softmax_pred = torch.zeros((temp_gt.shape[0], num_samples, batch_size))
        weighted_preds_pred = np.zeros((temp_gt.shape[0], batch_size, 51))

        # compute softmax with different temperatures and take average
        for ind, t in enumerate(temp_gt):
            score_ord_softmax_gt[ind] = F.softmax(t * torch.Tensor((score_ord_gt - score_ord_gt.max(0))), dim=0)
            weighted_preds_gt[ind] = (score_ord_softmax_gt[ind].unsqueeze(2).data.cpu().numpy() * outputs_samples_use).sum(axis=0)
            score_ord_softmax_pred[ind] = F.softmax(temp_pred[ind] * torch.Tensor((score_ord_pred - score_ord_pred.max(0))), dim=0)
            weighted_preds_pred[ind] = (score_ord_softmax_pred[ind].unsqueeze(2).data.cpu().numpy() * outputs_samples_use).sum(axis=0)

        # compute error statistics for the mini-batch
        sqerr = (outputs_use - targets_use) ** 2
        sqerr_samples = (outputs_samples_use - targets_use) ** 2
        sqerr_weighted_ord_gt = (weighted_preds_gt - targets_use) ** 2
        sqerr_weighted_ord_pred = (weighted_preds_pred - targets_use) ** 2

        distance = np.zeros((batch_size, 17))
        distance_samples = np.zeros((num_samples, batch_size, 17))
        distance_ord_weighted_gt = np.zeros((temp_pred.shape[0], batch_size, 17))
        distance_ord_weighted_pred = np.zeros((temp_gt.shape[0], batch_size, 17))

        dist_idx = 0
        for k in np.arange(0, 17 * 3, 3):
            distance[:, dist_idx] = np.sqrt(np.sum(sqerr[:, k:k + 3], axis=1))
            distance_samples[:, :, dist_idx] = np.sqrt(np.sum(sqerr_samples[:, :, k:k + 3], axis=2))
            for ind, t in enumerate(temp_gt):
                distance_ord_weighted_gt[ind, :, dist_idx] = np.sqrt(np.sum(sqerr_weighted_ord_gt[ind, :, k:k + 3], axis=1))
                distance_ord_weighted_pred[ind, :, dist_idx] = np.sqrt(np.sum(sqerr_weighted_ord_pred[ind, :, k:k + 3], axis=1))
            dist_idx += 1

        # append batch error statistics to global error trackers
        all_dist.append(distance)
        all_dist_samples.append(distance_samples)
        for ind, t in enumerate(temp_gt):
            all_dist_ordsamp_weighted[ind].append(distance_ord_weighted_gt[ind])
            all_dist_ordsamp_weighted_pred[ind].append(distance_ord_weighted_pred[ind])

        if (i % 10 == 0):
            print('({batch}/{size}) | loss: {loss:.6f}' \
                  .format(batch=i + 1,
                          size=len(test_loader),
                          loss=l2_loss.avg))
            sys.stdout.flush()

    # compute and report all error metrics
    ttl_err_mean = np.mean(np.vstack(all_dist))

    joint_err_samples = np.mean(np.concatenate(all_dist_samples, axis=1), axis=2)
    best_samples_err = np.min(joint_err_samples, axis=0)
    ttl_err_bestsamp = np.mean(best_samples_err)

    ttl_err_ord_weighted, ttl_err_ord_weighted_pred = {}, {}
    for ind, t in enumerate(temp_gt):
        ttl_err_ord_weighted[ind] = np.mean(np.vstack(all_dist_ordsamp_weighted[ind]))
        ttl_err_ord_weighted_pred[ind] = np.mean(np.vstack(all_dist_ordsamp_weighted_pred[ind]))
    ttl_err_ord_gt, best_temp_gt = np.min(np.array(list(ttl_err_ord_weighted.values()))), 0.1 * ( np.argmin(np.array(list(ttl_err_ord_weighted.values()))) + 1 )
    ttl_err_ord_pred, best_temp_pred = np.min(np.array(list(ttl_err_ord_weighted_pred.values()))), 0.1 * (np.argmin(np.array(list(ttl_err_ord_weighted_pred.values()))) + 1 )

    print("\n>>> Cumulative errors <<<")
    print(">>> Mean sample - {:4f} <<<".format(ttl_err_mean))
    print(">>> OrdinalScore ( PRED Ordinals ) - {:4f}, temp - {:.1f} <<<".format(ttl_err_ord_pred, best_temp_pred))
    print(">>> OrdinalScore ( GT Ordinals ) - {:4f}, temp - {:.1f} <<<".format(ttl_err_ord_gt, best_temp_gt))
    print(">>> Oracle - {:4f} <<<".format(ttl_err_bestsamp))

    return l2_loss.avg, ttl_err_mean, ttl_err_bestsamp, np.array(list(ttl_err_ord_weighted.values())), np.array(list(ttl_err_ord_weighted_pred.values()))

def main(opt):
    start_epoch = 0
    err_best = 1000
    glob_step = 0
    lr_now = opt.lr

    # save options
    log.save_options(opt, opt.ckpt)

    # create model
    print(">>> creating model")
    model = CVAE_Linear(opt.cvaeSize, opt.latent_size, opt.numSamples_train, opt.alpha, opt.cvae_num_stack)
    model.cuda()
    model.apply(weight_init)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    criterion = nn.MSELoss(size_average=True).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    # load ckpt
    if opt.load:
        print(">>> loading ckpt from '{}'".format(opt.load))
        ckpt = torch.load(opt.load)
        start_epoch = ckpt['epoch']
        err_best = ckpt['err']
        glob_step = ckpt['step']
        lr_now = ckpt['lr']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print(">>> ckpt loaded (epoch: {} | err: {})".format(start_epoch, err_best))
    if opt.resume:
        logger = log.Logger(os.path.join(opt.ckpt, 'log.txt'), resume=True)
    else:
        logger = log.Logger(os.path.join(opt.ckpt, 'log.txt'))
        logger.set_names(['epoch', 'lr', 'loss_train', 'loss_test', 'err_mean', 'err_bestsamp'])

    # list of action(s)
    actions = utils.define_actions('All')
    num_actions = len(actions)
    print(">>> actions to use (total: {}):".format(num_actions))
    pprint(actions, indent=4)
    print(">>>")

    # data loading
    print(">>> loading data")
    # load statistics data
    stat_2d = torch.load(os.path.join(opt.data_dir, 'stat_2d.pth.pt'))
    stat_3d = torch.load(os.path.join(opt.data_dir, 'stat_3d.pth.pt'))

    # test
    if opt.test:
        err_mean_set, err_bestsamp_set, err_ordsamp_weighted_set, err_ordsamp_weighted_set_pred = [], [], [], []

        for action in actions:
            print("\n>>> TEST on _{}_".format(action))
            test_loader = DataLoader(
                dataset=Human36M(actions=action, data_path=opt.data_dir, is_train=False, procrustes=opt.procrustes),
                batch_size=opt.test_batch,
                shuffle=False,
                num_workers=opt.job,
                pin_memory=True)

            _, err_mean_test, err_bestsamp_test, err_ordsamp_weighted_test, err_ordsamp_weighted_test_pred = test_multiposenet(test_loader, model, criterion, stat_3d, stat_2d, procrustes=opt.procrustes)

            err_mean_set.append(err_mean_test)
            err_bestsamp_set.append(err_bestsamp_test)
            err_ordsamp_weighted_set.append(err_ordsamp_weighted_test)
            err_ordsamp_weighted_set_pred.append(err_ordsamp_weighted_test_pred)

        err_ordsamp_weighted_set_all = np.stack(err_ordsamp_weighted_set, axis=1)
        err_ordsamp_weighted_set_pred_all = np.stack(err_ordsamp_weighted_set_pred, axis=1)
        err_ordsamp_weighted_set_all = np.mean(err_ordsamp_weighted_set_all, axis=1)
        err_ordsamp_weighted_set_pred_all = np.mean(err_ordsamp_weighted_set_pred_all, axis=1)

        best_temp_gt, best_val = np.argmin(err_ordsamp_weighted_set_all), np.min(err_ordsamp_weighted_set_all)
        best_temp_pred, best_val_pred = np.argmin(err_ordsamp_weighted_set_pred_all), np.min(err_ordsamp_weighted_set_pred_all)

        # print('Gt best temp : {:1f}, best val : {:.4f}'.format((best_temp_gt + 1) * 0.1, best_val))
        # print('Pred best temp : {:1f}, best val : {:.4f}'.format((best_temp_pred + 1) * 0.1, best_val_pred))

        err_ordsamp_weighted_set = np.stack(err_ordsamp_weighted_set, axis=1)[best_temp_gt]
        err_ordsamp_weighted_set_pred = np.stack(err_ordsamp_weighted_set_pred, axis=1)[best_temp_pred]

        print("\n\n>>>>>> TEST results:")
        for action in actions:
            print("{}".format(action), end='\t')
        print("\n")

        for err in err_mean_set:
            print("{:.4f}".format(err), end='\t')
        print(">>>\nERRORS - Mean : {:.4f}".format(np.array(err_mean_set).mean()))

        for err in err_ordsamp_weighted_set_pred:
            print("{:.4f}".format(err), end='\t')
        print(">>>\nERRORS - OrdinalScore ( PRED Ordinals ) : {:.4f}".format(np.array(err_ordsamp_weighted_set_pred).mean()))

        for err in err_ordsamp_weighted_set:
            print("{:.4f}".format(err), end='\t')
        print(">>>\nERRORS - OrdinalScore ( GT Ordinals ) : {:.4f}".format(np.array(err_ordsamp_weighted_set).mean()))

        for err in err_bestsamp_set:
            print("{:.4f}".format(err), end='\t')
        print(">>>\nERRORS - Oracle : {:.4f}".format(np.array(err_bestsamp_set).mean()))

        sys.exit()

    # load dadasets for training
    train_loader = DataLoader(
        dataset=Human36M(actions=actions, data_path=opt.data_dir, procrustes=opt.procrustes),
        batch_size=opt.train_batch,
        shuffle=True,
        num_workers=opt.job, )

    test_loader = DataLoader(
        dataset=Human36M(actions=actions, data_path=opt.data_dir, is_train=False, procrustes=opt.procrustes),
        batch_size=opt.test_batch,
        shuffle=False,
        num_workers=opt.job, )

    print(">>> data loaded !")

    cudnn.benchmark = True
    for epoch in range(start_epoch, opt.epochs):
        print('==========================')
        print('>>> epoch: {} | lr: {:.5f}'.format(epoch + 1, lr_now))

        glob_step, lr_now, loss_train = train_multiposenet(
            train_loader, model, criterion, optimizer,
            lr_init=opt.lr, lr_now=lr_now, glob_step=glob_step, lr_decay=opt.lr_decay, gamma=opt.lr_gamma,
            max_norm=opt.max_norm)
        loss_test, err_mean, err_bestsamp,_, _ = test_multiposenet(test_loader, model, criterion, stat_3d, stat_2d, procrustes=opt.procrustes)

        logger.append([epoch + 1, lr_now, loss_train, loss_test, err_mean, err_bestsamp],
                      ['int', 'float', 'float', 'float', 'float', 'float'])

        is_best = err_bestsamp < err_best
        err_best = min(err_bestsamp, err_best)
        if is_best:
            log.save_ckpt({'epoch': epoch + 1,
                           'lr': lr_now,
                           'step': glob_step,
                           'err': err_best,
                           'state_dict': model.state_dict(),
                           'optimizer': optimizer.state_dict()},
                          ckpt_path=opt.ckpt,
                          is_best=True)
        else:
            log.save_ckpt({'epoch': epoch + 1,
                           'lr': lr_now,
                           'step': glob_step,
                           'err': err_best,
                           'state_dict': model.state_dict(),
                           'optimizer': optimizer.state_dict()},
                          ckpt_path=opt.ckpt,
                          is_best=False)

    logger.close()

if __name__ == "__main__":
    option = Options().parse()
    main(option)
