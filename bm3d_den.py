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

class Solver():
    def __init__(self, conf):
        self.fmt = '\r[{:4d}/{:4d}] mse_loss={:.2f}, pnsr={:.2f}, ssim={:.2f}, z-mean={:.2f}, z-var={:.2f}'

        self.loader_dict = utils.get_celeb_data(
            img_size=conf.img_size, batch_size=conf.batch_size)
        
        self.test_loader = self.loader_dict['test_loader'][0]
        self.test_size = self.loader_dict['test_loader'][1]

        checkpoints_path = os.path.join(
                conf.root_path,
                'exp',
                'try-%d' % conf.try_num,
                'checkpoints'
        )

        if not os.path.exists(checkpoints_path):
            os.makedirs(checkpoints_path)

        self.device = conf.device
        self.model = model.FSGF(conf).to(self.device)
        summary(self.model)

        self.checkpoints_path = checkpoints_path
        self.load_epoch = conf.load_epoch
        self.conf = conf
        
        self._load_model()

    def solver(self, init_modes):
        for init_mode in init_modes:
            for it in range(6):
                print('solver in {:s}[{:d}]'.format(init_mode, it))
                self.solver_it(it, init_mode)

    def initialize(self, i_batch, init_mode):
        self.model.eval()

        self.i_batch = i_batch
        g_data  = iter(self.test_loader)
        for i, _ in zip(range(self.i_batch), g_data):
            pass

        x, _  = next(g_data)
        x_ = x.to(torch.float32)

        y  = torch.clamp(torch.floor(x_ + torch.randn(*x.size()) * 25.5), min=0, max=255)
        y  = y.to(torch.int32)

        x_deq, _ = modules.Dequantization.dequant(x)
        y_deq, _ = modules.Dequantization.dequant(y)

        self.x_deq = x_deq.detach().clone().to(self.device)
        self.y_deq = y_deq.detach().clone().to(self.device)

        self.init_mode  = init_mode
        self.mse_metric = torch.nn.MSELoss().to(self.device)            
        
    def solver_it(self, i_batch, init_mode):
        self.model.eval()
        self.initialize(i_batch, init_mode)
        
        res = []
        start_t = time.time()  
        
        img_x = modules.Dequantization.inv_dequant(self.x_deq).to(torch.float32).cpu()
        img_y = modules.Dequantization.inv_dequant(self.y_deq).to(torch.float32).cpu()

        for i in range(self.conf.iterations):
            assert self.init_mode == 'bm3d'
            
            x_hat_t = []
            for y_deq_i in self.y_deq:
                y_deq_i = y_deq_i.permute(1,2,0)
                y_norm = (y_deq_i * 0.5 + 0.5).detach().cpu().numpy()
                x_hat  = bm3d.bm3d_rgb(y_norm, sigma_psd=25/255)
                x_hat  = torch.tensor(x_hat, device=self.device).unsqueeze(0)
                x_hat_t.append(x_hat)

            x_hat_t = torch.cat(x_hat_t, dim=0).permute(0,3,1,2) * 2 - 1

            img_x_hat = modules.Dequantization.inv_dequant(x_hat_t.detach()).to(torch.float32).cpu()
            mse_sqrt  = torch.sqrt(self.mse_metric(img_x, img_x_hat))
            psnr = 20 * np.log10(255 / torch.sqrt(mse_sqrt))
            ssim_val = ssim(img_x, img_x_hat, data_range=255, size_average=True)

            print(self.fmt.format(
                i+1, self.conf.iterations, -1, psnr, ssim_val, -1, -1), end=" ")
            res.append([-1, psnr, ssim_val, -1, -1])
        print

        m = torch.cat([img_y, img_x, img_x_hat], dim=0)

        path = os.path.join('res','bm3d', str(self.i_batch), self.init_mode)
        if not os.path.exists(path):
            os.makedirs(path)

        utils.save_plots(m, self.conf, filename=os.path.join(path,"img_m.png"))
        utils.save_plots(img_y, self.conf, filename=os.path.join(path,"img_y.png"))
        utils.save_plots(img_x, self.conf, filename=os.path.join(path,"img_x.png"))
        utils.save_plots(img_x_hat, self.conf, filename=os.path.join(path,"img_x_hat.png"))

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
    parser.add_argument('--init', default='rand', help="inverse problem")

#   Hyperparameters
    parser.add_argument('--hid_dim', default=512, type=int, help="number of hidden channels") #256
    parser.add_argument('--K', default=2, type=int, help="K number")
    parser.add_argument('--L', default=5, type=int, help="L number")
    parser.add_argument('--batch_size', type=int, default=32 , help='batch size') #64
    parser.add_argument('--iterations', type=int, default=1, help='number of epochs')
    parser.add_argument('--model_lr', type=float, default=5e-3, help='learning rate in rec_loss.')

    conf = parser.parse_args()
    
    solver = Solver(conf)
    #solver.initialize(0, 'ours')
    #solver.solver_it()

    init_modes = ['bm3d']
    solver.solver(init_modes)