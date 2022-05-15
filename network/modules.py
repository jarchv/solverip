import torch
import torch.nn  as nn
import numpy as np
import scipy.linalg
import torch.nn.functional as F

from tools import utils

if torch.cuda.is_available(): 
    DEVICE = "cuda"
else: 
    DEVICE = "cpu"

def conv4x4(in_channels, out_channels, bias=True, stride=2, padding=1):
    return nn.Conv2d(
        in_channels=in_channels, out_channels=out_channels, stride=stride,
        kernel_size=4, bias=bias, padding=padding)

def conv3x3(in_channels, out_channels, bias=True, stride=1, padding=1):
    return nn.Conv2d(
        in_channels=in_channels, out_channels=out_channels, 
        kernel_size=3, bias=bias, padding=padding)

def conv2x2(in_channels, out_channels, bias=True, stride=1, padding=0):
    return nn.Conv2d(
        in_channels=in_channels, out_channels=out_channels, stride=stride,
        kernel_size=2, bias=bias, padding=padding)

def conv2x2_lpad(in_channels, out_channels, bias=True, stride=1, padding=0):
    return nn.Conv2d(
        in_channels=in_channels, out_channels=out_channels, stride=stride,
        kernel_size=2, bias=bias, padding=padding)
        
def conv1x1(in_channels, out_channels, bias=True, stride=1, padding=0):
    return nn.Conv2d(
        in_channels=in_channels, out_channels=out_channels, 
        kernel_size=1, bias=bias)

class Dequantization(nn.Module):
    def __init__(self, nbits=5):
        super().__init__()
        self.nbits  = nbits
        self.quants = 2 ** nbits

    @staticmethod
    def dequant(x=None, log_det=None, nbits=5):
        quants = 2 ** nbits
        x = x.to(torch.float32)
        x = torch.floor(x / 2 ** (8 - nbits))
        x = x + torch.rand_like(x).detach()
        x = x / quants
        x = 2 * x - 1

        if log_det != None:
            log_det -= np.log(256) * utils.count_pixels(x)
            log_det += (np.log(2) * utils.count_pixels(x))

        x = x.to(DEVICE)

        return x, log_det

    @staticmethod
    def inv_dequant(x=None, log_det=None, quants=32):
        x = x * 0.5 + 0.5
        x = x * quants
        x = torch.floor(x)
        x = x * 256. / quants
        x = torch.clamp(x, min=0, max=255).to(torch.int32)

        if log_det != None:
            log_det -= np.log(2) * utils.count_pixels(x)
            log_det += np.log(256) * utils.count_pixels(x)
        return x, log_det

    def forward(self, x, log_det=None, reverse=False):
        if not reverse:
            x, log_det = self.dequant(x, log_det, self.nbits)
            return x, log_det
        else:
            x, log_det = self.inv_dequant(x, log_det, self.quants)
            return x, log_det

class ActNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bias  = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.log_s = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.initialized = False
        self.dim = dim

    def initialize(self, x, eps=1e-6):
        if not self.training:
            return
            
        with torch.no_grad():
            mean = utils.reduce_mean(x.clone(), dim=[0,2,3], keepdim=True)
            var  = utils.reduce_mean((x.clone() - mean)**2, dim=[0,2,3], keepdim=True)
            bias  = -mean
            log_s = torch.log(1./(torch.sqrt(var) + eps))

            self.bias.data.copy_(bias.data)
            self.log_s.data.copy_(log_s.data)
            self.initialized = True

    def _center(self, x, reverse=False):
        if not reverse:
            x = x + self.bias
            return x
        else:
            return x - self.bias

    def _scale(self, x, log_det=None, reverse=False):
        dlog_det = utils.reduce_sum(self.log_s) * utils.count_pixels(x)
        if not reverse:
            x = x * torch.exp(self.log_s)
            if log_det != None:
                log_det = log_det + dlog_det
            return x, log_det
        else:
            log_det = log_det - dlog_det
            x = x * torch.exp(-self.log_s)
            return x, log_det
    
    def forward(self, x, log_det=None, reverse=False):       
        if not self.initialized:
            self.initialize(x)
        if not reverse:
            x = self._center(x, reverse)
            x, log_det = self._scale(x, log_det, reverse)
            return x, log_det
        else:
            x, log_det = self._scale(x, log_det, reverse)
            x = self._center(x, reverse)
            return x, log_det

class Invertible1x1Conv2D(nn.Module):
    def __init__(self, dim, LU_decomp=False):
        super().__init__()
        w_shape = (dim, dim)
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)

        if not LU_decomp:
            self.W = nn.Parameter(torch.Tensor(w_init))
        else:
            raise NotImplementedError()
        self.w_shape = w_shape
    
    def forward(self, x, log_det=None, reverse=False):
        if not reverse:
            W = self.W.view(*self.w_shape, 1, 1)
            x = F.conv2d(x, W)

            p_factor = utils.count_pixels(x)
            log_det += (torch.log(torch.abs(torch.det(self.W))) * p_factor)

            return x, log_det
        else:
            W = torch.inverse(self.W.double()).float().view(*self.w_shape,1,1)
            x = F.conv2d(x, W)

            return x

class NN(nn.Module):
    conv_type = {1: (conv1x1, 0, 1), 2: (conv2x2, 0, 4), 4: (conv3x3, 0, 4), 8: (conv3x3, 1, 1)}
    def __init__(self, in_channels, hid_channels, out_channels, wsize):
        super().__init__()
        
        conv, pad, ch_factor = self.conv_type[min(wsize, 8)]

        hid_channels_mod = hid_channels * ch_factor
        out_channels_mod = out_channels * ch_factor

        self.conv1 = conv(in_channels, hid_channels_mod, padding=pad)
        self.norm1 = ActNorm(hid_channels)
        self.conv2 = conv(hid_channels, out_channels_mod, padding=pad)
        
        self.up_factor = int(ch_factor ** 0.5)
        self.log_s = nn.Parameter(torch.zeros(out_channels, 1, 1))

        nn.init.normal_(self.conv1.weight, mean=0, std=0.01)
        nn.init.zeros_(self.conv1.bias)

        self.conv2.weight.data.zero_()
        self.conv2.bias.data.zero_()


    def forward(self, x):
        x = self.conv1(x)
        x = utils.unsqueeze2d(x=x, factor=self.up_factor)

        x = F.relu(self.norm1(x)[0])
        x = self.conv2(x) 
        
        x = utils.unsqueeze2d(x=x, factor=self.up_factor)
        x = x * torch.exp(3. * self.log_s)

        return x    

class AffineCouplingLayer(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, swap, wsize):
        super().__init__()

        self.NN = NN(in_channels // 2, hid_channels, out_channels, wsize)
        self.swap = swap

    def forward(self, x, log_det, reverse):
        x1, x2 = utils.split_channels(x, mode='chunk', swap=self.swap)
        s_and_t = self.NN(x2)
        s, t = utils.split_channels(s_and_t, mode='cross')
        s = torch.sigmoid(s + 2.)
        dlog_det = utils.reduce_sum(torch.log(s), dim=[1,2,3])
        if not reverse:
            log_det = log_det + dlog_det

            y1 = x1 * s + t      
            y_ = [y1, x2]

            if self.swap:
                y_ = y_[::-1]
            y = torch.cat(y_, dim=1)

            return y, log_det

        else:
            y1 = (x1 - t) / s
            y_ = [y1, x2]

            if self.swap:
                y_ = y_[::-1]
            y = torch.cat(y_, dim=1)

            log_det = log_det - dlog_det
            return y, log_det

class SqueezeLayer(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, x, log_det=None, reverse=False):
        if not reverse:
            y = utils.squeeze2d(x, self.factor)
            return y, log_det
        else:
            y = utils.unsqueeze2d(x, self.factor)
            return y, log_det

class RandomSplit2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = Conv2dWithZeros(dim // 2, dim)

    def split2d_prior(self, x):
        h = self.conv(x)
        return utils.split_channels(h, "cross")

    def forward(self, x, log_det=None, reverse=False):
        if not reverse:
            z1, z2 = utils.split_channels(x, "chunk")
            mean, log_sigma = self.split2d_prior(z1)
            log_det += Gaussian.log_prob(mean, log_sigma, z2)
            return z1, log_det
        else:
            z1 = x  
            mean, log_s = self.split2d_prior(z1)
            z2 = Gaussian.sample(mean.size(),mean, log_s).to(DEVICE)
            z = torch.cat([z1, z2], dim=1)
            return z

class Split2d(nn.Module):
    def __init__(self, C, H, W):
        super().__init__()

    def forward(self, x, y=None, log_det=None, reverse=False):
        if not reverse:
            x1, x2 = utils.split_channels(x, "chunk")
            return x1, x2
        else:
            assert y is not None
            x = torch.cat([x, y], dim=1)
            return x, log_det

class Gaussian:
    log2PI = float(np.log(2 * np.pi))

    @staticmethod
    def flatten_sum(tensor):
        if len(tensor.shape) == 2:
            return utils.reduce_sum(tensor, dim=1)
        elif len(tensor.shape) == 4:
            return utils.reduce_sum(tensor, dim=[1, 2, 3])

    @staticmethod
    def _log_likelihood(mean, log_s, z):
        return -0.5 * (Gaussian.log2PI + 2. * log_s + ((z - mean) ** 2) / torch.exp(2. * log_s))

    @staticmethod
    def log_prob(mean, log_s, z):
        ll = Gaussian._log_likelihood(mean, log_s, z)
        return Gaussian.flatten_sum(ll)

    @staticmethod
    def sample(t_shape, mean=None, log_s=None):
        eps = torch.randn(t_shape).to(DEVICE)
        if mean is None:
            return eps
        return mean + torch.exp(log_s) * eps