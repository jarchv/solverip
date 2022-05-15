import torch
import torch.nn as nn
import numpy as np

from network import modules
from tools import utils

if torch.cuda.is_available(): 
    DEVICE = "cuda"
else: 
    DEVICE = "cpu"

class FlowStep(nn.Module):
    def __init__(self, in_channels, hid_channels, swap, H):
        super().__init__()
        self.norm = modules.ActNorm(in_channels)
        #self.perm = modules.Invertible1x1Conv2D(in_channels)
        self.layer = modules.AffineCouplingLayer(
            in_channels, hid_channels, in_channels, swap, H)
    
    def forward(self, x, log_det=None, reverse=False):
        if not reverse:
            x, log_det = self.norm(x, log_det, reverse)
            #x, log_det = self.perm(x, log_det, reverse)
            x, log_det = self.layer(x, log_det, reverse)

            return x, log_det
        else:
            x, log_det = self.layer(x, log_det, reverse)
            #x = self.perm(x, log_det, reverse)
            x, log_det = self.norm(x, log_det, reverse)
            return x, log_det

class FSGF(nn.Module):
    def __init__(self, conf):
        super().__init__()
        input_shape  = (conf.img_ch, conf.img_size, conf.img_size)
        hid_dim = conf.hid_dim
        K = conf.K
        L = conf.L

        C, H, W = input_shape
        self.layers = nn.ModuleList()
        self.layers.append(modules.Dequantization())

        sizes = []
        swap = False
        Km = K
        for i in range(L):
            self.layers.append(modules.SqueezeLayer(factor=2))
            C, H, W = C * 4, H // 2, W // 2

            assert H == W
    
            # Inv 1x1 Conv
            # if H > 4:
            #    Km = 4
            #else:
            #    Km = 6

            # SWAP
            if H >= 4:
               Km = K
            else:
               Km = K * 2
            
            # L = 5
            for _ in range(Km):
                self.layers.append(FlowStep(C, hid_dim, swap, H))
                
                # Inv 1x1 Conv
                # assert L == 3
                
                # SWAP
                assert L == 5
                swap = not swap

            if H <= 4 and i < L - 1:
                self.layers.append(modules.Split2d(C, H, W))
                C = C // 2
                sizes.append([C, H, W])   
        sizes.append([C, H, W])
        self.sizes = sizes
        self.C = C
        self.fixed_sample = False

    def g(self, Z, log_det, deq=True):
        z = Z[-1]
        features = reversed(Z[:-1])
        for layer in reversed(self.layers):
            if layer.__class__.__name__ == "Split2d":
                y = next(features)
                z, log_det = layer(z, y, log_det, reverse=True)
            elif layer.__class__.__name__ == "Dequantization":
                if deq == True:
                    z, log_det = layer(z, log_det, reverse=True)
            else:
                z, log_det = layer(z, log_det, reverse=True)
        return z, log_det

    def f(self, x, log_det, deq=True):
        Z = []
        for layer in self.layers:
            if layer.__class__.__name__ == "Split2d":
                x, z = layer(x, log_det, reverse=False)
                Z.append(z)
            elif layer.__class__.__name__ == "Dequantization":
                if deq == True:
                    x, log_det = layer(x, log_det, reverse=False)
            else:
                x, log_det = layer(x, log_det, reverse=False)
        Z.append(x)
        return Z, log_det

    def set_actnorm_init(self):
        for m in self.layers:
            if m.__class__.__name__ == 'FlowStep':
                m.norm.initialized = True
                m.layer.NN.norm1.initialized = True

    def compute_loss(self, z, log_det):
        mean  = torch.zeros_like(z)
        log_s = torch.zeros_like(z)
        
        log_prob_z = modules.Gaussian.log_prob(mean=mean, log_s=log_s, z=z)
        return -torch.mean(log_prob_z + log_det)   

    def criterion(self, x, deq=True):
        B = x.size(0)
        Z, log_det = self(x, reverse=False, deq=deq)
        z = self.flatten_Z(Z)

        loss = self.compute_loss(z, log_det)
        return loss, z

    def sampling(self, bsize, fixed=True):
        if not self.fixed_sample or fixed == False:
            t_shape = (bsize, *self.sizes)
            self.Z = [modules.Gaussian.sample(t_shape=(bsize, *s)).to(DEVICE)*1.8 for s in self.sizes]
            self.fixed_sample = not self.fixed_sample
        log_det = torch.zeros(bsize).to(DEVICE)
        return self.g(self.Z, log_det)  

    def flatten_Z(self, Z):
        B = Z[0].size(0)
        z = [z.view(B,-1) for z in Z]
        z = torch.cat(z, dim=1)
        return z

    def unflatten_z(self, z):
        splid_elements = [np.prod(s) for s in self.sizes]
        Z = []
        
        start = 0
        for n, size in zip(splid_elements, self.sizes):
            end = start + n
            z_  = z[:,start:end].view([-1]+list(size))
            Z.append(z_)
            start = end
        return Z

    def forward(self, x, log_det=None, reverse=False, deq=True):
        if not reverse:
            bsize = x.size(0)
            log_det = torch.zeros(bsize).to(DEVICE)
            Z, log_det = self.f(x=x, log_det=log_det, deq=deq)
            return Z, log_det
        else:
            if log_det == None:
                bsize = x[0].size(0)
                log_det = torch.zeros(bsize).to(DEVICE)
            x_hat, log_det = self.g(x, log_det, deq=deq)
            return x_hat, log_det

class Clipper(object):
    def __init__(self):
        pass
    def __call__(self, module):
        if module.__class__.__name__ == 'AffineCouplingLayerWithMask':
            w = module.scaling_factor
            print(w.min().item(), w.max().item())
            #w.clamp_(torch.norm(w, 2, 1).expand_as(w))