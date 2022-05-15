import torch
import numpy as np
import torch.nn.functional as F

from network import modules

if torch.cuda.is_available(): 
    DEVICE = "cuda"
else: 
    DEVICE = "cpu"

class Inpainting():
    def __init__(self, test_loader, i_batch, init_mode):
        self.g = iter(test_loader)
        self.i = i_batch

    def create_mask(self, x):
        B,C,H,W = x.size()

        mask = np.ones([H,W])
        l = int(H * (0.5 - 0.2))
        r = int(W * (0.5 + 0.2))
        mask[l:r, l:r] = 0.
        mask = mask.reshape(1,1,H,W)
        mask = torch.tensor(mask, dtype=torch.float)

        self.mask = mask.to(DEVICE)

    def measures(self, xf, train=False, noise=None):
        y    = xf * self.mask
        return y

    def makedata(self):
        for i, _ in zip(range(self.i), self.g):
            pass

        x, _  = next(self.g)
        xf = x.to(torch.float32).to(DEVICE)

        self.create_mask(xf)
        y  = self.measures(xf)
        y  = y.to(torch.int32)

        x_deq = modules.Dequantization.dequant(x)[0].clone()
        y_deq = modules.Dequantization.dequant(y)[0].clone()

        return x_deq, y_deq

class Deblurring():
    def __init__(self, test_loader, i_batch, init_mode):
        self.g = iter(test_loader)
        self.i = i_batch

    def measures(self, xf, noise=None):
        y = F.avg_pool2d(xf, kernel_size=3, stride=1, padding=1)
        return y

    def makedata(self):
        for i, _ in zip(range(self.i), self.g):
            pass
        x, _  = next(self.g)

        xf = x.to(torch.float32).to(DEVICE)

        x_deq = modules.Dequantization.dequant(x)[0].clone()
        y_deq = self.measures(x_deq)
        
        return x_deq, y_deq

class Denoising():
    def __init__(self, test_loader, i_batch, init_mode):
        self.g = iter(test_loader)
        self.i = i_batch

    def measures(self, xf):
        return xf

    def noisy_measures(self, xf,):
        y = torch.clamp(torch.floor(xf + self.noise), min=0, max=255)
        return y

    def makedata(self):
        for i, _ in zip(range(self.i), self.g):
            pass
        x, _  = next(self.g)

        xf = x.to(torch.float32).to(DEVICE)
        self.noise = torch.randn(*xf.size(), device=DEVICE) * 25.5

        y  = self.noisy_measures(xf)
        y  = y.to(torch.int32)

        x_deq = modules.Dequantization.dequant(x)[0].clone()
        y_deq = modules.Dequantization.dequant(y)[0].clone()

        return x_deq, y_deq

class Colorization():
    def __init__(self, test_loader, i_batch, init_mode):
        self.g = iter(test_loader)
        self.i = i_batch

    def measures(self, xf, noise=None):
        y  = xf.mean(dim=1, keepdim=True).repeat(1,3,1,1)
        return y

    def makedata(self):
        for i, _ in zip(range(self.i), self.g):
            pass
        x, _  = next(self.g)

        xf = x.to(torch.float32).to(DEVICE)
        y  = self.measures(xf)
        y  = y.to(torch.int32)

        x_deq = modules.Dequantization.dequant(x)[0].clone()
        y_deq = modules.Dequantization.dequant(y)[0].clone()

        return x_deq, y_deq