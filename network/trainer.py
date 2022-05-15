import time
import os
import torch
import numpy as np
import itertools
import matplotlib.pyplot as plt

from tools import utils
from network import model
from torchsummary import summary

class Trainer():
    def __init__(self, conf):
        self.fmt_train = '\rEpoch {:4d}[{:5d}/{:5d}] loss={:.2f}, mean={:.2f}, var={:.2f}'
        self.fmt_valid = ', loss(val)={:.2f} in {:.1f} sec.'

        self.loader_dict = utils.get_celeb_data(
            img_size=conf.img_size, batch_size=conf.batch_size)
        
        self.train_load = self.loader_dict['train_loader'][0]
        self.train_size = self.loader_dict['train_loader'][1]
        self.valid_load = self.loader_dict['valid_loader'][0]

        self.max_grad_clip = 5      # fixed
        self.max_grad_norm = 100    # fixed

        checkpoints_path = os.path.join(
                conf.root_path,
                'exp',
                'try-%d' % conf.try_num,
                'checkpoints')

        if not os.path.exists(checkpoints_path):
            os.makedirs(checkpoints_path)

        self.device = conf.device
        self.model = model.FSGF(conf).to(self.device)
        self.opt_model = torch.optim.Adam(
            itertools.chain(self.model.parameters()), lr=conf.model_lr, betas=(conf.beta1, 0.999))
        summary(self.model)

        self.checkpoints_path = checkpoints_path
        self.load_epoch = conf.load_epoch
        self.conf = conf
        
        self._load_model()

        plt.ion()
        self.fig = plt.figure(figsize=(5, 6), dpi=80)
        plt.show(block=False)

    def show_inputs(self):
        for image_batch, _ in self.train_load:
            z, log_det = self.model(image_batch, reverse=False)
            print(log_det)
            x_hat, log_det = self.model(z, log_det, reverse=True)
            print(log_det)
            utils.plot_inputs_iteractive(image_batch, self.conf, self.fig, "input")
            utils.plot_inputs_iteractive(x_hat, self.conf, self.fig, 'projecting z')    

    def sample_batch(self):
        self.model.eval()
        x_hat = self.model.sampling(16, fixed=True)[0].cpu().detach()
        utils.plot_inputs(x_hat, self.conf)    

    def train_step(self):
        x_hat = self.model.sampling(32, fixed=False)[0].cpu().detach()
        utils.plot_inputs_iteractive(x_hat, self.conf, self.fig) 

        for ep in range(self.load_epoch + 1, self.conf.epochs + 1):
            start_t = time.time()
            self.model.train()
            train_loss = []
            mean_list = []
            var_list = []

            it = 0
            for image_batch, _ in self.train_load:
                loss, mean, var = self.optimize_parameters(x=image_batch)

                it += image_batch.size(0)

                train_loss.append(loss)
                mean_list.append(mean)
                var_list.append(var)

                print_it = self.fmt_train.format(
                    ep, 
                    it,
                    self.train_size,
                    train_loss[-1],
                    mean, var)

                print(print_it, end='')
                self._save_model(ep)
                break
            train_loss_mean = np.mean(train_loss)
            print_train = self.fmt_train.format(
                ep, 
                it,
                self.train_size,
                train_loss_mean,
                np.mean(mean_list),
                np.mean(var_list))

            print("{:s}".format(print_train), end='')
            
            self.model.eval()
            valid_loss = []

            for image_batch_v, _ in self.valid_load:
                loss, *rest = self.model.criterion(
                    x=image_batch_v)
                valid_loss.append(loss.item())
            valid_loss_mean = np.mean(valid_loss)   

            end_t = time.time()        
            print_valid = self.fmt_valid.format(
                valid_loss_mean, end_t-start_t)
            print('{:s}{:s}'.format(print_train, print_valid)) 

            # Show sample
            x_hat = self.model.sampling(32, fixed=False).cpu().detach()
            utils.plot_inputs_iteractive(x_hat, self.conf, self.fig)  
            
            # Save model each {conf.save_freq} epochs
            if ep % self.conf.save_freq == 0: 
                self._save_model(ep)
                
    def _save_model(self, ep):
        print('Saving "model-{:d}"... '.format(ep), end='')

        file_model = 'model-{:d}.pth'.format(ep)
        save_path  = os.path.join(self.checkpoints_path, file_model)
		
        checkpoint = {}

        checkpoint['state_dict_net'] = self.model.state_dict()   
        checkpoint['opt_model'] = self.opt_model.state_dict()

        torch.save(checkpoint, save_path)
        print("Done.")

    def _load_model(self):
        if self.load_epoch <= 0:
            return
        print('\nLoading "model-{:d}"...'.format(self.load_epoch), end='')
        file_model = 'model-{:d}.pth'.format(self.load_epoch)

        load_path  = os.path.join(self.checkpoints_path, file_model)
        checkpoint = torch.load(load_path)

        self.model.load_state_dict(checkpoint['state_dict_net'])
        self.opt_model.load_state_dict(checkpoint['opt_model'])
        self.model.set_actnorm_init()

        print("Done.")

    def optimize_parameters(self, x):
        loss, z = self.model.criterion(x)
        self.opt_model.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.model.parameters(), self.max_grad_clip)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.opt_model.step()
        return loss.item(), z.mean().item(), z.var().item()