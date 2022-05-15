import time
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import pickle as pkl
from tools import utils
from network import model
from torchsummary import summary
from network import modules
from pytorch_msssim import ssim

import argparse
import itertools
import bm3d

import invprob

def norm2(x, y, z=None, log_det=None, gamma=None):
    bsize = x.size(0)
    return torch.square(x - y).view(bsize,-1).sum(dim=1).mean()

def norm1(x, y, z=None, log_det=None, gamma=None):
    bsize = x.size(0)
    return torch.abs(x - y).view(bsize,-1).sum(dim=1).mean()

def norm2_gamma(x, y, z=None, log_det=None, gamma=None):
    bsize = x.size(0)
    L     = (z.norm(dim=1)**2).mean() 
    return torch.square(x - y).view(bsize,-1).sum(dim=1).mean() + gamma * L

def bm3d_denoiser(y_deq, device):
    x_hat_ibatch = []
    for y_deq_i in y_deq:
        y_deq_i = y_deq_i.permute(1,2,0)
        y_norm = (y_deq_i * 0.5 + 0.5).detach().cpu().numpy()
        x_hat  = bm3d.bm3d_rgb(y_norm, sigma_psd=25/255)
        x_hat  = torch.tensor(x_hat, device=device).unsqueeze(0)
        x_hat_ibatch.append(x_hat)
    x_hat_t = torch.cat(x_hat_ibatch, dim=0).permute(0,3,1,2) * 2 - 1
    return x_hat_t

def criterion(x, y, z=None, log_det=None, gamma=None):
    bsize, zdim = z.size()
    mean  = torch.zeros_like(z)
    log_s = torch.zeros_like(z) 
    
    log_pz = modules.Gaussian.log_prob(mean=mean, log_s=log_s, z=z)
    log_px = torch.mean(log_pz - log_det)
    l1norm = torch.abs(x - y).view(bsize,-1).sum(dim=1).mean()

    alpha  = gamma
    loss   = -log_px * alpha + l1norm #alpha: 0.05(denoising), 0.002(inpainting), 0.02(deblu,color)
    return loss

def map_loss(x, y, z=None, log_det=None, gamma=None, prob_x=None):
    bsize, *rest = x.size()
    
    l2norm_term = norm2(x,y) / (2 * 0.1 ** 2)

    loss = l2norm_term - prob_x * 0.1
    return loss

class Solver():
    def __init__(self, conf):
        self.fmt = '\r[{:4d}/{:4d}] loss={:.2f}, pnsr={:.2f}, ssim={:.3f}, z-mean={:.2f}, z-var={:.2f}'
        self.loader_dict = utils.get_celeb_data(
            img_size=conf.img_size, batch_size=conf.batch_size)
        
        self.test_loader = self.loader_dict['test_loader'][0]
        self.test_size   = self.loader_dict['test_loader'][1]

        checkpoints_path = os.path.join(conf.root_path,'exp','try-%d' % conf.try_num,'checkpoints')

        if not os.path.exists(checkpoints_path):
            os.makedirs(checkpoints_path)

        self.device = conf.device
        self.model = model.FSGF(conf).to(self.device)

        summary(self.model)

        self.checkpoints_path = checkpoints_path
        self.load_epoch = conf.load_epoch
        self.conf = conf
        
        self._load_model()

    def solver(self, method_list):
        for method in method_list:
            for it in range(6):
                print('{:s} in {:s}[{:d}]'.format(self.conf.invprob, method, it))
                self.solver_it(it, method)

    def initialize(self, i_batch, method):
        self.model.eval()

        if self.conf.invprob == "denoising":
            self.invprob = invprob.Denoising(self.test_loader, i_batch, method)
        elif self.conf.invprob == "deblurring":
            self.invprob = invprob.Deblurring(self.test_loader, i_batch, method)
        elif self.conf.invprob == "inpainting":
            self.invprob = invprob.Inpainting(self.test_loader, i_batch, method)
        elif self.conf.invprob == "colorization":  
            self.invprob = invprob.Colorization(self.test_loader, i_batch, method)

        self.x_deq, self.y_deq = self.invprob.makedata()
        

        Z, log_det = self.model(torch.zeros_like(self.x_deq), reverse=False)
        z_flatten  = self.model.flatten_Z(Z)
        
        eps = torch.randn(*z_flatten.size(), device=self.device)
        if method == 'rand':
            self.gamma = 0
            z_flatten_init = eps
            self.z_flatten = z_flatten_init.detach().requires_grad_(True)
            self.criterion = norm2

        if method == 'rand-gamma':
            self.gamma = 0.1 # best: 0.01 (inpaiting, colorization, deblurring), 0.1 (denoising)
            self.z_flatten = z_flatten_init.detach().requires_grad_(True)
            self.criterion = norm2_gamma

        elif method == 'zeros':
            self.gamma = 0
            z_flatten_init = eps * 0
            self.z_flatten = z_flatten_init.detach().requires_grad_(True)
            self.criterion = norm2

        elif method == 'zeros-gamma':
            self.gamma = 0.1  # 0.05 (paper), 0.1 (best)
            z_flatten_init = eps * 0 
            self.z_flatten = z_flatten_init.detach().requires_grad_(True)
            self.criterion = norm2_gamma

        elif method == 'criterion':
            self.gamma = 0
            sigma = 0.1 # default (0.1)
            z_flatten_init = eps * sigma
            self.z_flatten = z_flatten_init.detach().requires_grad_(True)
            self.criterion = criterion

        elif method == 'map':
            self.gamma = 0
            sigma = 0.1
            z_flatten_init = eps * sigma
            self.z_flatten = z_flatten_init.detach().requires_grad_(True)
            self.criterion = map_loss

        #self.optimizer = torch.optim.LBFGS([self.z_flatten], lr=conf.model_lr)
        self.optimizer  = torch.optim.Adam(itertools.chain([self.z_flatten]), lr=conf.model_lr)
        self.mse_metric = torch.nn.MSELoss().to(self.device)      

        self.method  = method
        self.i_batch = i_batch

    def computer_metrics(self, img_pred, img_targ):
        mse_sqrt  = torch.sqrt(self.mse_metric(img_pred, img_targ))
        psnr_val = 20 * np.log10(255 / torch.sqrt(mse_sqrt))
        ssim_val = ssim(img_pred, img_targ, data_range=255, size_average=True)

        return psnr_val, ssim_val

    def solver_it(self, i_batch, init_mode):
        self.model.eval()
        self.initialize(i_batch, init_mode)
        
        res = []
        start_t = time.time()  
        
        img_targ = modules.Dequantization.inv_dequant(self.x_deq)[0].to(torch.float32).cpu()
        img_meas = modules.Dequantization.inv_dequant(self.y_deq)[0].to(torch.float32).cpu()

        bsize, zdim = self.z_flatten.size()

        path = os.path.join('res',self.conf.invprob, str(self.i_batch), self.method)
        if not os.path.exists(path):
            os.makedirs(path)

        time_list = []
        for i in range(self.conf.iterations):
            def closure():
                tic = time.time()
                
                self.optimizer.zero_grad()

                Z_init = self.model.unflatten_z(self.z_flatten)
                x_hat, log_det  = self.model(Z_init, reverse=True, deq=False)

                x_msr = self.invprob.measures(x_hat)

                if init_mode == "map":
                    prob_x_hat, z_new = self.model.criterion(x_hat, deq=False)
                    loss = self.criterion(x_msr, self.y_deq, self.z_flatten, log_det, self.gamma, prob_x_hat)
                else:
                    loss = self.criterion(x_msr, self.y_deq, self.z_flatten, log_det, self.gamma)
                loss.backward()

                img_pred = modules.Dequantization.inv_dequant(x_hat.detach())[0].to(torch.float32).cpu()

                time_list.append(time.time()-tic)

                psnr_val, ssim_val = self.computer_metrics(img_pred, img_targ)   
                
                
                print(self.fmt.format(
                    i+1, 
                    self.conf.iterations, 
                    loss.item(), 
                    psnr_val, 
                    ssim_val, 
                    self.z_flatten.mean().item(), 
                    self.z_flatten.var().item()), end=" ")
                
                res.append([loss.item(), 
                    psnr_val, 
                    ssim_val, 
                    self.z_flatten.mean().item(), 
                    self.z_flatten.var().item()])

                return loss

            self.optimizer.step(closure=closure)
        print
        #print("Frame Per Second: {:.2f}".format(( len(time_list) * self.z_flatten.size(0) / sum(time_list))))

        
        Z_init   = self.model.unflatten_z(self.z_flatten)
        x_hat, _ = self.model(Z_init, reverse=True, deq=False)
            
        img_pred = modules.Dequantization.inv_dequant(x_hat.detach())[0].to(torch.float32).cpu()

        m = torch.cat([img_meas, img_targ, img_pred], dim=0)

        utils.save_plots(m, self.conf, filename=os.path.join(path,"img_m.png"))
        utils.save_plots(img_meas, self.conf, filename=os.path.join(path,"img_meas.png"))
        utils.save_plots(img_targ, self.conf, filename=os.path.join(path,"img_targ.png"))
        utils.save_plots(img_pred, self.conf, filename=os.path.join(path,"img_pred.png"))

        with open(os.path.join(path,'iterations.pkl'), 'wb') as f:
            pkl.dump(res, f)

        end_t = time.time()     
        print("in {:.1f} sec.".format(end_t-start_t))
        

    def _load_model(self):
        if self.load_epoch <= 0:
            return
        print('\nLoading "model-{:d}"...'.format(self.load_epoch), end='')
        file_model = 'model-{:d}.pth'.format(self.load_epoch)

        load_path  = os.path.join(self.checkpoints_path, file_model)
        checkpoint = torch.load(load_path)

        self.model.load_state_dict(checkpoint['state_dict_net'])
        self.model.set_actnorm_init()

        print("Done.")

if __name__ == '__main__':

#   Dataset
    parser = argparse.ArgumentParser(description="Train Network")
    parser.add_argument('--root_path', default='./', help='root path for cheackpoints')
    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument('--img_size', type=int, default=32, help='image size')
    parser.add_argument('--img_ch', type=int, default=3, help='image channels')

#   Experiments
    parser.add_argument('--save_freq', type=int, default=10, help='save after every "save_freq" epoch')
    parser.add_argument('--load_epoch', type=int, default=100,help='load at "load_epoch" epoch')
    parser.add_argument('--try_num', default=1, type=int, help="try number")
    parser.add_argument('--invprob', default='inpainting', help="inverse problem")
    parser.add_argument('--method', default='rand', help="method")

#   Hyperparameters
    parser.add_argument('--hid_dim', default=512, type=int, help="number of hidden channels") #256
    parser.add_argument('--K', default=2, type=int, help="K number")
    parser.add_argument('--L', default=5, type=int, help="L number")
    parser.add_argument('--batch_size', type=int, default=32 , help='batch size') #64
    parser.add_argument('--iterations', type=int, default=1500, help='number of epochs')
    parser.add_argument('--model_lr', type=float, default=5e-3, help='learning rate in rec_loss.')

    conf = parser.parse_args()
    
    solver = Solver(conf)
    #solver.initialize(0, 'ours')
    #solver.solver_it()

    #init_modes = ['rand', 'zeros', 'zeros-gamma']
    init_modes = ['map']
    solver.solver(init_modes)