#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

H36M_NAMES = ['']*32
H36M_NAMES[0]  = 'Hip'
H36M_NAMES[1]  = 'RHip'
H36M_NAMES[2]  = 'RKnee'
H36M_NAMES[3]  = 'RFoot'
H36M_NAMES[6]  = 'LHip'
H36M_NAMES[7]  = 'LKnee'
H36M_NAMES[8]  = 'LFoot'
H36M_NAMES[12] = 'Spine'
H36M_NAMES[13] = 'Thorax'
H36M_NAMES[14] = 'Neck/Nose'
H36M_NAMES[15] = 'Head'
H36M_NAMES[17] = 'LShoulder'
H36M_NAMES[18] = 'LElbow'
H36M_NAMES[19] = 'LWrist'
H36M_NAMES[25] = 'RShoulder'
H36M_NAMES[26] = 'RElbow'
H36M_NAMES[27] = 'RWrist'

SH_NAMES = ['']*16
SH_NAMES[0]  = 'RFoot'
SH_NAMES[1]  = 'RKnee'
SH_NAMES[2]  = 'RHip'
SH_NAMES[3]  = 'LHip'
SH_NAMES[4]  = 'LKnee'
SH_NAMES[5]  = 'LFoot'
SH_NAMES[6]  = 'Hip'
SH_NAMES[7]  = 'Spine'
SH_NAMES[8]  = 'Thorax'
SH_NAMES[9]  = 'Head'
SH_NAMES[10] = 'RWrist'
SH_NAMES[11] = 'RElbow'
SH_NAMES[12] = 'RShoulder'
SH_NAMES[13] = 'LShoulder'
SH_NAMES[14] = 'LElbow'
SH_NAMES[15] = 'LWrist'

def define_actions(action):

    actions = ["Directions",
               "Discussion",
               "Eating",
               "Greeting",
                "Phoning",
               "Photo",
               "Posing",
               "Purchases",
                "Sitting",
               "SittingDown",
               "Smoking",
               "Waiting",
               "WalkDog",
               "Walking",
               "WalkTogether"]

    if action == "All" or action == "all":
        return actions

    if action not in actions:
        raise (ValueError, "Unincluded action: {}".format(action))

    return [action]

class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def lr_decay(optimizer, step, lr, decay_step, gamma):
    lr = lr * gamma ** (step/decay_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def postproc(ord):
    batch_size = ord.shape[0]
    proc_ord = ord.copy()
    # setting diagonal of ordinal matrix to the equality constraint
    for j in range(16):
        proc_ord[:, j, j] = 2
    return proc_ord

def compute_ordinals(pts, threshold=1):

    pts_temp= pts.copy()

    if (len(pts_temp.shape) == 3):
        numSamples, batch_size = pts_temp.shape[0], pts_temp.shape[1]

        ordinal_array = np.zeros([numSamples, batch_size, 16, 16]).astype(np.int32)
        ordinal_array.fill(2)

        ge = np.greater_equal(np.tile(pts_temp[:, :, None, :], [1, 1, 16, 1]), np.tile(pts_temp[:, :, :, None], [1, 1, 1, 16]) + threshold)
        le = np.less_equal(np.tile(pts_temp[:, :, None, :], [1, 1, 16, 1]) + threshold, np.tile(pts_temp[:, :, :, None], [1, 1, 1, 16]))

        ordinal_array[ge] = 0
        ordinal_array[le] = 1

        return ordinal_array
    else:
        batch_size = pts_temp.shape[0]

        ordinal_array = np.zeros([batch_size,16,16]).astype(np.int32)
        ordinal_array.fill(2)

        ge = np.greater_equal(np.tile(pts_temp[:,None,:], [1,16,1]), np.tile(pts_temp[:,:,None], [1,1,16]) + threshold)
        le = np.less_equal(np.tile(pts_temp[:,None,:], [1,16,1]) + threshold, np.tile(pts_temp[:,:,None], [1,1,16]))

        ordinal_array[ge] = 0
        ordinal_array[le] = 1

        return ordinal_array

def compare(samp_ord, comp_ord):
    num_samp, batch_size = samp_ord.shape[0], samp_ord.shape[1]
    tmp_ord = comp_ord.copy()
    tmp_ord = np.tile(tmp_ord[None], [num_samp, 1, 1, 1])

    # computing inconsistent entries in pred ordinal matrix
    tmp_ord_inv = tmp_ord.copy()
    tmp_ord_inv = np.transpose(tmp_ord_inv, [0, 1, 3, 2])
    m0, m1 = tmp_ord_inv == 0, tmp_ord_inv == 1
    tmp_ord_inv[m0] = 1
    tmp_ord_inv[m1] = 0
    mask = np.not_equal(tmp_ord_inv, tmp_ord)

    score_ord = np.equal(samp_ord, tmp_ord).astype(np.int32)  # comparing ordinal entries
    score_ord[mask] = 0  # zeroing out the inconsistent entries
    score_ord = score_ord.sum((-2, -1))

    return score_ord

def get_transformation(X, Y, compute_optimal_scale=False):
    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0 ** 2.).sum()
    ssY = (Y0 ** 2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 = X0 / normX
    Y0 = Y0 / normY

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    # Make sure we have a rotation
    detT = np.linalg.det(T)
    V[:, -1] *= np.sign(detT)
    s[-1] *= np.sign(detT)
    T = np.dot(V, U.T)

    traceTA = s.sum()

    if compute_optimal_scale:  # Compute optimum scaling of Y.
        b = traceTA * normX / normY
        d = 1 - traceTA ** 2
        Z = normX * traceTA * np.dot(Y0, T) + muX
    else:  # If no scaling allowed
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    c = muX - b * np.dot(muY, T)

    return d, Z, T, b, c

def unNormalizeData(normalized_data, data_mean, data_std, dimensions_to_use):
    T = normalized_data.shape[0]  # Batch size
    D = data_mean.shape[0]  # 96

    orig_data = np.zeros((T, D), dtype=np.float32)

    orig_data[:, dimensions_to_use] = normalized_data

    # Multiply times stdev and add the mean
    stdMat = data_std.reshape((1, D))
    stdMat = np.repeat(stdMat, T, axis=0)
    meanMat = data_mean.reshape((1, D))
    meanMat = np.repeat(meanMat, T, axis=0)
    orig_data = np.multiply(orig_data, stdMat) + meanMat
    return orig_data