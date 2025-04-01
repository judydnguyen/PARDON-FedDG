from collections import OrderedDict
from numbers import Number
import time
import warnings

import logging
import operator
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import src.functions as functions
# from function import adaIN_StyleStat_ContentFeat, coral

from torchvision import transforms

logger = logging.getLogger(__name__)

#########################
# AdaIN utils #
#########################

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    
    size = feat.shape
    assert (len(size) == 4)
    N, C, _, _ = size
    feat = feat.swapaxes(1,0)

    feat_var = feat.reshape(C, -1).var(axis=1) + eps
    feat_std = torch.sqrt(feat_var.reshape(1, C, 1, 1))
    feat_mean = feat.reshape(C, -1).mean(axis=1).reshape(1, C, 1, 1)
    # import pdb; pdb.set_trace() # feat:[1, 512, 87, 64], feat_mean&std:[1, 512, 1, 1], 
    return feat_mean, feat_std

def calc_sum(feat):
    feat = feat.detach()
    size = feat.shape
    assert (len(size) == 4)
    N, C, H, W = size
    count = N * H * W
    feat = feat.swapaxes(1,0)
    
    feat_sum = feat.reshape(C, -1).sum(axis=1).reshape(1, C, 1, 1)
    feat_square = feat ** 2
    feat_square_sum = feat_square.reshape(C, -1).sum(axis=1).reshape(1, C, 1, 1)
    # import pdb; pdb.set_trace() 
    return feat_sum, feat_square_sum, count

def style_transfer(vgg, decoder, content, style_stat, alpha=1.0,
                   interpolation_weights=None, device="cuda"):
    assert (0.0 <= alpha <= 1.0)
    content = content.to(device)
    content_f = vgg(content)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = functions.adaIN_StyleStat_ContentFeat(content_f, style_stat)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = functions.adaIN_StyleStat_ContentFeat(content_f, style_stat)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat.to(device))

#########################
# Weight initialization #
#########################

def init_weights(model, init_type, init_gain):
    """Function for initializing network weights.
    
    Args:
        model: A torch.nn instance to be initialized.
        init_type: Name of an initialization method (normal | xavier | kaiming | orthogonal).
        init_gain: Scaling factor for (normal | xavier | orthogonal).
    
    Reference:
        https://github.com/DS3Lab/forest-prediction/blob/master/pix2pix/models/networks.py
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            else:
                raise NotImplementedError(f'[ERROR] ...initialization method [{init_type}] is not implemented!')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        
        elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)   
    model.apply(init_func)

def init_net(model, init_type, init_gain, gpu_ids):
    """Function for initializing network weights.
    
    Args:
        model: A torch.nn.Module to be initialized
        init_type: Name of an initialization method (normal | xavier | kaiming | orthogonal)l
        init_gain: Scaling factor for (normal | xavier | orthogonal).
        gpu_ids: List or int indicating which GPU(s) the network runs on. (e.g., [0, 1, 2], 0)
    
    Returns:
        An initialized torch.nn.Module instance.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        model.to(gpu_ids[0])
        model = nn.DataParallel(model, gpu_ids)
    init_weights(model, init_type, init_gain)
    return model    

def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

class ParamDict(OrderedDict):
    """Code adapted from https://github.com/Alok/rl_implementations/tree/master/reptile.
    A dictionary where the values are Tensors, meant to represent weights of
    a model. This subclass lets you perform arithmetic on weights directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def sign(self):
        return ParamDict({k: torch.sign(v) for k, v in self.items()})
    
    def ge(self, number):
        return ParamDict({k: torch.ge(v, number) for k, v in self.items()})
    
    def le(self, number):
        return ParamDict({k: torch.le(v, number) for k, v in self.items()})
    
    def gt(self, number):
        return ParamDict({k: torch.gt(v, number) for k, v in self.items()})
    
    def lt(self, number):
        return ParamDict({k: torch.lt(v, number) for k, v in self.items()})
    
    def abs(self):
         return ParamDict({k: torch.abs(v) for k, v in self.items()})

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            raise NotImplementedError

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        # a- b := a + (-b)
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)

    def to(self, device):
        return ParamDict({k: v.to(device) for k, v in self.items()})

## Copied from https://github.com/ildoonet/pytorch-gradual-warmup-lr/blob/master/warmup_scheduler/scheduler.py
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class GradualWarmupScheduler(_LRScheduler):

    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, total_epoch, init_lr=1e-7, after_scheduler=None):
        self.init_lr = init_lr
        assert init_lr > 0, 'Initial LR should be greater than 0.'
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.finished = True
                return self.after_scheduler.get_lr()
            return self.base_lrs

        return [(((base_lr - self.init_lr) / self.total_epoch) * self.last_epoch + self.init_lr) for base_lr in
                self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [(((base_lr - self.init_lr) / self.total_epoch) * self.last_epoch + self.init_lr) for base_lr in
                         self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if (self.finished and self.after_scheduler) or self.total_epoch == 0:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


#########################
# Redundant #
#########################

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def calc_sum_domains(feat, total_domains, metadata):
    feat_sums, feat_square_sums, counts = [], [], []
    for domain in range(total_domains):
        idx_list = torch.where(metadata[:,0]==domain)
        if idx_list:
            feat_sum, feat_square_sum, count = calc_sum(feat[idx_list[0].numpy()])
        else:
            feat_sum, feat_square_sum, count = [], [], 0
        feat_sums.append(feat_sum)
        feat_square_sums.append(feat_square_sum)
        counts.append(count)
    return feat_sums, feat_square_sums, counts        

def calc_style_loss(input, target):
    mse_loss = nn.MSELoss()
    assert (input.size() == target.size())
    # assert (input.requires_grad is False)
    input_mean, input_std = calc_mean_std(input)
    target_mean, target_std = calc_mean_std(target)
    return mse_loss(input_mean, target_mean) + \
            mse_loss(input_std, target_std)
            
def normalize(x):
    # Normalize x (input or target) here, for example, by subtracting mean and dividing by standard deviation
    normalized_x = (x - x.mean()) / x.std()
    return normalized_x

def calc_content_loss(normalized_input, normalized_target):
    mse_loss = nn.MSELoss()
    assert (normalized_input.size() == normalized_target.size())
    # assert (normalized_input.requires_grad is False)
    
    return mse_loss(normalized_input, normalized_target)
    
# Function to compute Gram matrix
def gram_matrix(features):
    _, c, h, w = features.size()
    import IPython
    IPython.embed()
    features = features.view(c, h * w)
    gram = torch.mm(features, features.t())
    return gram.div(c * h * w)

def gram_matrix_batch(features):
    # Get the dimensions of the input features
    batch_size, c, h, w = features.size()

    # Reshape the features to have a 2D shape (batch_size * c, h * w)
    features = features.view(batch_size, c, h * w)

    # Compute the Gram matrix for each element in the batch
    gram_matrices = torch.bmm(features, features.transpose(1, 2))

    # Normalize the Gram matrices by the number of elements (h * w)
    gram_matrices = gram_matrices.div(c * h * w)

    return gram_matrices

# Function to compute the style difference between two images
def style_difference(input_features, target_features, layer_name='conv4_2'):

    # Get the desired layer for style representation
    input_style = input_features
    target_style = target_features

    # Compute Gram matrix for style representation
    input_gram = gram_matrix_batch(input_style)
    target_gram = gram_matrix_batch(target_style)

    # Compute the Frobenius norm (L2 norm of the difference) between Gram matrices
    difference = torch.norm((input_gram - target_gram))/input_features.shape[0]

    return difference.item()

class TookTooLong(Warning):
    pass

class MinimizeStopper(object):
    def __call__(self, xk=None, convergence = 0):
        self.max_sec = 120
        self.start = time.time()
    def __call__(self, xk=None):
        elapsed = time.time() - self.start
        if elapsed > self.max_sec:
            warnings.warn("Terminating optimization: time limit reached",
                          TookTooLong)
        else:
            # you might want to report other stuff here
            print("Elapsed: %.3f sec" % elapsed)
            
    """
    N-Pair loss
    Sohn, Kihyuk. "Improved Deep Metric Learning with Multi-class N-pair Loss Objective," Advances in Neural Information
    Processing Systems. 2016.
    http://papers.nips.cc/paper/6199-improved-deep-metric-learning-with-multi-class-n-pair-loss-objective
    """

    def __init__(self, l2_reg=0.01):
        super(NPairLoss2, self).__init__()
        self.l2_reg = l2_reg

    def forward(self, embeddings, transferred_embeddings, target):
        n_pairs, n_negatives = self.get_n_pairs(target)

        if embeddings.is_cuda:
            n_pairs = n_pairs.cuda()
            n_negatives = n_negatives.cuda()
        # print(f"n_pairs: {n_pairs}")
        # print(f"n_negatives: {n_negatives}")
        # anchors = embeddings    # (n, embedding_size)
        # positives = transferred_embeddings
        # positives = embeddings[n_pairs[:, 1]]  # (n, embedding_size)
        negatives = embeddings[n_negatives]    # (n, n-1, embedding_size)
        negatives_2 = transferred_embeddings[n_negatives]

        losses = self.n_pair_loss(embeddings, transferred_embeddings, negatives) \
            + self.l2_reg * self.l2_loss(embeddings, transferred_embeddings)    \
            + self.n_pair_loss(embeddings, transferred_embeddings, negatives_2)
        return losses

    @staticmethod
    def get_n_pairs(labels):
        """
        Get index of n-pairs and n-negatives
        :param labels: label vector of mini-batch
        :return: A tuple of n_pairs (n, 2)
                        and n_negatives (n, n-1)
        """
        labels = labels.cpu().data.numpy()
        n_pairs = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            anchor, positive = np.random.choice(label_indices, 2, replace=False)
            n_pairs.append([anchor, positive])

        n_pairs = np.array(n_pairs)

        n_negatives = []
        idxs = [i for i in range(labels.shape[0])]
        for i in range(len(idxs)):
            negative = np.concatenate([idxs[:i], idxs[i+1:]])
            n_negatives.append(negative)

        n_negatives = np.array(n_negatives)

        return torch.LongTensor(n_pairs), torch.LongTensor(n_negatives)

    @staticmethod
    def n_pair_loss(anchors, positives, negatives):
        """
        Calculates N-Pair loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :param negatives: A torch.Tensor, (n, n-1, embedding_size)
        :return: A scalar
        """
        anchors = torch.unsqueeze(anchors, dim=1)  # (n, 1, embedding_size)
        positives = torch.unsqueeze(positives, dim=1)  # (n, 1, embedding_size)

        x = torch.matmul(anchors, (negatives - positives).transpose(1, 2))  # (n, 1, n-1)
        x = torch.sum(torch.exp(x), 2)  # (n, 1)
        loss = torch.mean(torch.log(1+x))
        return loss

    @staticmethod
    def l2_loss(anchors, positives):
        """
        Calculates L2 norm regularization loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :return: A scalar
        """
        return torch.sum(anchors ** 2 + positives ** 2) / anchors.shape[0]
        # return torch.sum(anchors ** 2) / anchors.shape[0]