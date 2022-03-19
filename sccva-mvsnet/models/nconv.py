########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
# this file is modified from nconv and aerial-depth-completion repo
########################################

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.conv import _ConvNd
import numpy as np
import torch.nn as nn
from scipy.stats import poisson
from scipy import signal
import math
try:
    from torch._six import container_abcs
except ImportError:
    import collections.abc as container_abcs
from itertools import repeat


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


# The proposed Normalized Convolution Layer
class NConv2d(_ConvNd):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 pos_fn='softplus',
                 init_method='k',
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros'):

        # Call _ConvNd constructor
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(NConv2d, self).__init__(in_channels,
                                      out_channels,
                                      kernel_size,
                                      stride,
                                      padding,
                                      dilation,
                                      False,
                                      0,
                                      groups,
                                      bias,
                                      padding_mode)

        self.eps = 1e-20
        self.pos_fn = pos_fn
        self.init_method = init_method

        # Initialize weights and bias
        self.init_parameters()

        if self.pos_fn is not None:
            EnforcePos.apply(self, 'weight', pos_fn)

    def forward(self, data, conf):

        # Normalized Convolution
        denom = F.conv2d(conf, self.weight, None, self.stride,
                         self.padding, self.dilation, self.groups)
        nomin = F.conv2d(data * conf, self.weight, None, self.stride,
                         self.padding, self.dilation, self.groups)
        nconv = nomin / (denom + self.eps)

        # Add bias
        b = self.bias
        sz = b.size(0)
        b = b.view(1, sz, 1, 1)
        b = b.expand_as(nconv)
        nconv += b

        # Propagate confidence
        cout = denom
        sz = cout.size()
        cout = cout.view(sz[0], sz[1], -1)

        k = self.weight
        k_sz = k.size()
        k = k.view(k_sz[0], -1)
        s = torch.sum(k, dim=-1, keepdim=True)

        cout = cout / s
        cout = cout.view(sz)

        return nconv, cout

    def init_parameters(self):
        # Init weights
        if self.init_method == 'x':  # Xavier
            torch.nn.init.xavier_uniform_(self.weight)
        elif self.init_method == 'k':  # Kaiming
            torch.nn.init.kaiming_uniform_(self.weight)
        elif self.init_method == 'n':  # Normal dist
            n = self.kernel_size[0] * self.kernel_size[1] * self.out_channels
            self.weight.data.normal_(2, math.sqrt(2. / n))
        elif self.init_method == 'p':  # Poisson
            mu = self.kernel_size[0] / 2
            dist = poisson(mu)
            x = np.arange(0, self.kernel_size[0])
            y = np.expand_dims(dist.pmf(x), 1)
            w = signal.convolve2d(y, y.transpose(), 'full')
            w = torch.Tensor(w).type_as(self.weight)
            w = torch.unsqueeze(w, 0)
            w = torch.unsqueeze(w, 1)
            w = w.repeat(self.out_channels, 1, 1, 1)
            w = w.repeat(1, self.in_channels, 1, 1)
            self.weight.data = w + torch.rand(w.shape)
        else:
            print('Undefined Initialization method!')
            return

            # Init bias
        self.bias = torch.nn.Parameter(torch.zeros(self.out_channels) + 0.01)


# Non-negativity enforcement class
class EnforcePos(object):
    def __init__(self, pos_fn, name):
        self.name = name
        self.pos_fn = pos_fn

    @staticmethod
    def apply(module, name, pos_fn):
        fn = EnforcePos(pos_fn, name)

        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, inputs):
        if module.training:
            weight = getattr(module, self.name)
            weight.data = self._pos(weight).data
        else:
            pass

    def _pos(self, p):
        pos_fn = self.pos_fn.lower()
        if pos_fn == 'softmax':
            p_sz = p.size()
            p = p.view(p_sz[0], p_sz[1], -1)
            p = F.softmax(p, -1)
            return p.view(p_sz)
        elif pos_fn == 'exp':
            return torch.exp(p)
        elif pos_fn == 'softplus':
            return F.softplus(p, beta=10)
        elif pos_fn == 'sigmoid':
            return F.sigmoid(p)
        else:
            print('Undefined positive function!')
            return


class SC2UnguidedDense(nn.Module):
    # SparseConfidence2UnguidedDense
    def __init__(self, pos_fn='SoftPlus', num_channels=2):
        super().__init__()

        self.pos_fn = pos_fn

        self.nconv1 = NConv2d(1, num_channels, (5, 5), pos_fn, 'p', padding=2)
        self.nconv2 = NConv2d(num_channels, num_channels, (5, 5), pos_fn, 'p', padding=2)
        self.nconv3 = NConv2d(num_channels, num_channels, (5, 5), pos_fn, 'p', padding=2)

        self.nconv4 = NConv2d(2 * num_channels, num_channels, (3, 3), pos_fn, 'p', padding=1)
        self.nconv5 = NConv2d(2 * num_channels, num_channels, (3, 3), pos_fn, 'p', padding=1)
        self.nconv6 = NConv2d(2 * num_channels, num_channels, (3, 3), pos_fn, 'p', padding=1)

        self.nconv7 = NConv2d(num_channels, 1, (1, 1), pos_fn, 'k')

    def forward(self, x0, c0):

        x1, c1 = self.nconv1(x0, c0)
        x1, c1 = self.nconv2(x1, c1)
        x1, c1 = self.nconv3(x1, c1)

        # Downsample 1
        ds = 2
        c1_ds, idx = F.max_pool2d(c1, ds, ds, return_indices=True)
        x1_ds = torch.zeros_like(c1_ds)
        for i in range(x1_ds.size(0)):
            for j in range(x1_ds.size(1)):
                x1_ds[i, j, :, :] = x1[i, j, :, :].view(-1)[idx[i, j, :, :].view(-1)].view(idx.size()[2:])
        c1_ds /= 4

        x2_ds, c2_ds = self.nconv2(x1_ds, c1_ds)
        x2_ds, c2_ds = self.nconv3(x2_ds, c2_ds)

        # Downsample 2
        ds = 2
        c2_dss, idx = F.max_pool2d(c2_ds, ds, ds, return_indices=True)

        x2_dss = torch.zeros_like(c2_dss)
        for i in range(x2_dss.size(0)):
            for j in range(x2_dss.size(1)):
                x2_dss[i, j, :, :] = x2_ds[i, j, :, :].view(-1)[idx[i, j, :, :].view(-1)].view(idx.size()[2:])
        c2_dss /= 4

        x3_ds, c3_ds = self.nconv2(x2_dss, c2_dss)

        # Downsample 3
        ds = 2
        c3_dss, idx = F.max_pool2d(c3_ds, ds, ds, return_indices=True)

        x3_dss = torch.zeros_like(c3_dss)
        for i in range(x3_dss.size(0)):
            for j in range(x3_dss.size(1)):
                x3_dss[i, j, :, :] = x3_ds[i, j, :, :].view(-1)[idx[i, j, :, :].view(-1)].view(idx.size()[2:])
        c3_dss /= 4
        x4_ds, c4_ds = self.nconv2(x3_dss, c3_dss)

        # Upsample 1
        x4 = F.interpolate(x4_ds, c3_ds.size()[2:], mode='nearest')
        c4 = F.interpolate(c4_ds, c3_ds.size()[2:], mode='nearest')
        x34_ds, c34_ds = self.nconv4(torch.cat((x3_ds, x4), 1), torch.cat((c3_ds, c4), 1))

        # Upsample 2
        x34 = F.interpolate(x34_ds, c2_ds.size()[2:], mode='nearest')
        c34 = F.interpolate(c34_ds, c2_ds.size()[2:], mode='nearest')
        x23_ds, c23_ds = self.nconv5(torch.cat((x2_ds, x34), 1), torch.cat((c2_ds, c34), 1))

        # Upsample 3
        x23 = F.interpolate(x23_ds, x0.size()[2:], mode='nearest')
        c23 = F.interpolate(c23_ds, c0.size()[2:], mode='nearest')
        xout, cout = self.nconv6(torch.cat((x23, x1), 1), torch.cat((c23, c1), 1))

        xout, cout = self.nconv7(xout, cout)

        # xout = torch.cat((xout,cout), 1)

        return xout, cout
