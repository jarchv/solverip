import argparse
from network import trainer


if __name__ == '__main__':

#   Dataset
    parser = argparse.ArgumentParser(description="Train Network")
    parser.add_argument('--root_path', default='./', help='root path for cheackpoints')
    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument('--img_size', type=int, default=32, help='image size')                      # fixed
    parser.add_argument('--img_ch', type=int, default=3, help='image channels')                     # fixed

#   Experiments
    parser.add_argument('--save_freq', type=int, default=5, help='save after every "save_freq" epoch')
    parser.add_argument('--load_epoch', type=int, default=100,help='load at "load_epoch" epoch')
    parser.add_argument('--try_num', default=1, type=int, help="try number")

#   Hyperparameters
    parser.add_argument('--hid_dim', default=512, type=int, help="number of hidden channels")       # fixed
    parser.add_argument('--K', default=2, type=int, help="K number")
    parser.add_argument('--L', default=5, type=int, help="L number")
    parser.add_argument('--batch_size', type=int, default=32 , help='batch size')                   # fixed
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--model_lr', type=float, default=1e-4, help='learning rate in rec_loss.')  # fixed
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')        # fixed

    conf = parser.parse_args()
    
    trainer = trainer.Trainer(conf)
    #trainer.sample_batch()
    #trainer.show_inputs()
    trainer.train_step()