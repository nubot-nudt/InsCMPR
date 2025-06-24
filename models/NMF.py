# -*- coding: utf-8 -*-
# @Time    : 2020/06/24 16:46
# @Author  : Xu HongBin
# @Email   : 2775751197@qq.com or 17770026885@163.com
# @github  : https://github.com/ToughStoneX
# @blog    : https://blog.csdn.net/hongbin_xu
# @File    : seg_dff
# @Software: PyCharm

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import random


EPSILON = 1e-7

# object function for nmf
def approximation_error(V, W, H, square_root=True):
    # Frobenius Norm
    return torch.norm(V - torch.mm(W, H))


def multiplicative_update_step(V, W, H, update_h=None, VH=None, HH=None):
    # update operation for W
    if VH is None:
        assert HH is None
        Ht = torch.t(H)  # [k, m] --> [m, k]
        VH = torch.mm(V, Ht)  # [n, m] x [m, k] --> [n, k]
        HH = torch.mm(H, Ht)  # [k, m] x [m, k] --> [k, k]

    WHH = torch.mm(W, HH) # [n, k] x [k, k] --> [n, k]
    WHH[WHH == 0] = EPSILON
    W *= VH / WHH

    if update_h:
        # update operation for H (after updating W)
        Wt = torch.t(W)  # [n, k] --> [k, n]
        WV = torch.mm(Wt, V)  # [k, n] x [n, m] --> [k, m]
        WWH = torch.mm(torch.mm(Wt, W), H)  #
        WWH[WWH == 0] = EPSILON
        H *= WV / WWH
        VH, HH = None, None

    return W, H, VH, HH


def NMF(V, k, W=None, H=None, random_seed=None, max_iter=200, tol=1e-4, cuda=True, verbose=False):
    if verbose:
        start_time = time.time()

    scale = torch.sqrt(V.mean() / k)

    if random_seed is not None:
        if cuda:
            current_random_seed = torch.cuda.initial_seed()
            torch.cuda.manual_seed(random_seed)
        else:
            current_random_seed = torch.initial_seed()
            torch.manual_seed(random_seed)

    if W is None:
        if cuda:
            W = torch.cuda.FloatTensor(V.size(0), k).normal_()
        else:
            W = torch.randn(V.size(0), k)
        W *= scale  # [n, k]

    update_H = True
    if H is None:
        if cuda:
            H = torch.cuda.FloatTensor(k, V.size(1)).normal_()
        else:
            H = torch.randn(k, V.size(1))
        H *= scale  # [k, m]
    else:
        update_H = False

    if random_seed is not None:
        if cuda:
            torch.cuda.manual_seed(current_random_seed)
        else:
            torch.manual_seed(current_random_seed)

    W = torch.abs(W)
    H = torch.abs(H)

    error_at_init = approximation_error(V, W, H, square_root=True)
    previous_error = error_at_init

    VH = None
    HH = None
    for n_iter in range(max_iter):
        W, H, VH, HH = multiplicative_update_step(V, W, H, update_h=update_H, VH=VH, HH=HH)
        if tol > 0 and n_iter % 10 == 0:
            error = approximation_error(V, W, H, square_root=True)
            if (previous_error - error) / error_at_init < tol:
                break
            previous_error = error

    if verbose:
        print('Exited after {} iterations. Total time: {} seconds'.format(n_iter+1, time.time()-start_time))
    return W, H




