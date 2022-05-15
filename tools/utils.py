import torch
import torchvision
import argparse
import matplotlib.pyplot as plt
import numpy as np
import time

from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.utils import make_grid

def reduce_mean(x, dim=None, keepdim=False):
    if dim is None:
        return torch.mean(x)

    if isinstance(dim, int):
        dim = [dim]
    
    dim = sorted(dim)
    for d in dim:
        x = torch.mean(x, dim=d, keepdim=True)

    if not keepdim:
        for cnt, d in enumerate(dim):
            x.squeeze_(d - cnt)
    return x

def reduce_sum(x, dim=None, keepdim=False):
    if dim is None:
        return torch.sum(x)
    
    if isinstance(dim, int):
        dim = [dim]

    dim = sorted(dim)
    for d in dim:
        x = torch.sum(x, dim=d, keepdim=True)

    if not keepdim:
        for cnt, d in enumerate(dim):
            x.squeeze_(d - cnt)
    return x

def count_pixels(x):
    assert len(x.size()) == 4
    return x.size(2) * x.size(3)
    
def squeeze2d(x, factor=2):
    if factor == 1: return x

    B, C, H, W = x.size()

    x = x.view(B, C, H // factor, factor, W // factor, factor)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(B, C * factor * factor, H // factor, W // factor)
    return x

def unsqueeze2d(x, factor=2):
    if factor == 1: return x
    
    B, C, H, W = x.size()
    x = x.view(B, C // (factor * factor), factor, factor, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(B, C // (factor * factor), H * factor, W * factor)
    return x

def save_plots(data, conf, vmax=255, filename=None):
    plt.figure(figsize=(18, 5), dpi=80)

    plt.xticks([])
    plt.yticks([])

    m = data.detach().cpu()

    m = make_grid(m, padding=1, nrow=conf.batch_size, pad_value=100)
    m = m.permute(1,2,0)
    m = np.uint8(m.to(torch.int32).cpu().numpy())
    print(filename)

    plt.imsave(filename, m, vmin=0, vmax=vmax)
    plt.close()
    
def plot_inputs(data, conf, vmax=255, title='Batch'):
    plt.figure(figsize=(18, 5), dpi=80)

    plt.xticks([])
    plt.yticks([])
    
    m = data.detach().cpu()
    m = make_grid(m, padding=1, nrow=conf.batch_size, pad_value=100)
    m = m.permute(1,2,0)

    #cmap = 'viridis'
    cmap = 'gray'
    if conf.img_ch == 1:
        m = m[:,:,0]

    plt.imshow(m.to(torch.int32), vmin=0, vmax=vmax)
    plt.title(title)
    plt.show()

    m = np.uint8(m.to(torch.int32).cpu().numpy())
    plt.imsave("batch.png", m, vmin=0, vmax=vmax)
    plt.close()

def plot_inputs_iteractive(data, conf, fig=None, title='Batch'):
    plt.xticks([])
    plt.yticks([])

    m = data.detach().cpu()
    #if m.max() <= 1:
    #    m = torch.clamp(torch.floor(m * 255), min=0, max=255).to(torch.int32)
    m = make_grid(m, padding=1, nrow=4, pad_value=100)
    m = m.permute(1,2,0)

    #cmap = 'viridis'
    cmap = 'gray'
    if conf.img_ch == 1:
        m = m[:,:,0]
        plt.imshow(m, cmap=cmap, vmin=0, vmax=255)
    else:
        plt.imshow(m, vmin=0, vmax=255)
    plt.title(title)
    time.sleep(1)
    fig.canvas.draw()
    plt.clf()
    #for axi in self.axs.flatten(): axi.clear()
    #plt.show()
    #plt.close()

def discretize(sample):
    return (sample * 255).to(torch.int32)

def split_channels(x, mode='chunk', swap=False):
    if mode == 'chunk':
        if swap:
            return x[:,x.size(1)//2:,:,:], x[:,:x.size(1)//2,:,:]
        return x[:,:x.size(1)//2,:,:], x[:,x.size(1)//2:,:,:]
    elif mode == 'cross':
        return x[:,0::2,:,:], x[:,1::2,:,:]

def create_checkerboard_mask(C, H, W):
    x = torch.arange(H, dtype=torch.int32)
    y = torch.arange(W, dtype=torch.int32)

    xx, yy = torch.meshgrid(x, y)
    mask = torch.fmod(xx + yy, 2)
    mask = mask.to(torch.float32).view(1,1,H,W)
    
    res = mask
    return res
    for ic in range(C-1):
        mask = 1 - mask
        res = torch.cat([res, mask], dim=1)
    return res

# DATASETS: MINST, TOY-SAMPLES, CELEBA
# ====================================

def get_mnist_data(test=False, batch_size=32):
    torch.manual_seed(7)
    
    transform = transforms.Compose([
        transforms.Pad(2,2),
        transforms.ToTensor(),
        discretize])
    data_dict = {}
    if test:
        test_set = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform)

        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=False)

        data_dict['test_loader'] = (test_loader, len(test_set))
        return data_dict    

    train_raw = torchvision.datasets.MNIST(
        root='./data', train=True, 
        download=True, transform=transform)

    train_set, valid_set = random_split(train_raw, [55000, 5000])
    train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(
            valid_set, batch_size=batch_size, shuffle=False)
            
    data_dict['train_loader'] = train_loader, len(train_set)
    data_dict['valid_loader'] = valid_loader, len(valid_set)

    return data_dict

def get_celeb_data(img_size=48, batch_size=4):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        #transforms.Grayscale(num_output_channels=1),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        discretize])
    celeb_dataset = torchvision.datasets.ImageFolder(
        root ='./data',
        transform = transform)

    indices = list(range(0, len(celeb_dataset)))
    train_idx = indices[:162770]
    valid_idx = indices[162770:182637]
    test_idx  = indices[182637:] 

    train_set = torch.utils.data.Subset(celeb_dataset, train_idx)
    valid_set = torch.utils.data.Subset(celeb_dataset, valid_idx)
    test_set  = torch.utils.data.Subset(celeb_dataset, test_idx)

    train_size = len(train_set)
    valid_size = len(valid_set)
    test_size  = len(test_set)
    
    assert (train_size + valid_size + test_size) == len(celeb_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=batch_size, shuffle=False)

    test_loader  = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False)

    data_dict = {}
    data_dict['train_loader'] = (train_loader, train_size)
    data_dict['valid_loader'] = (valid_loader, valid_size)
    data_dict['test_loader']  = (test_loader , test_size )

    return data_dict

class ToyData(Dataset):
    def __init__(self, train=True, transform=None):
        self.train = train
        
        m1 = np.array([[0, 1], [0, 1]], dtype=np.float32)
        m2 = np.array([[1, 0], [1, 0]], dtype=np.float32)
        m3 = np.array([[1, 1], [0, 0]], dtype=np.float32)
        m4 = np.array([[0, 0], [1, 1]], dtype=np.float32)

        self.data = [m1, m2, m3, m4]
        self.length = len(self.data)
        self.t = transform
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, i):
        return self.t(self.data[i]), -1

def get_toy_data(img_size=4, batch_size=4):
    torch.manual_seed(7)  
    transform = transforms.Compose([
        transforms.ToTensor(),
        discretize])
    data_dict = {}

    
    train_set = ToyData(transform=transform)
    valid_set = ToyData(transform=transform)

    train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(
            valid_set, batch_size=batch_size, shuffle=False)
            
    data_dict['train_loader'] = train_loader, len(train_set)
    data_dict['valid_loader'] = valid_loader, len(valid_set)

    return data_dict