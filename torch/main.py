import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import os
import sys
import time

from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from torchvision.utils import save_image

from model import CRNN, weights_init

sys.path.append('..')
from tools.dataset import lmdbDataset
from tools.utlis import srtLabelConverter

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs',
                    type=int,
                    default=20,
                    help='Number of training epoch. Default: 20')
parser.add_argument('--batch_size',
                    type=int,
                    default=128,
                    help='The number of batch_size. Default: 32')
parser.add_argument('--learning_rate',
                    type=float,
                    default=1e-3,
                    help='Learning rate during optimization. Default: 1e-3')
parser.add_argument('--data_dir',
                    default='../data/lmdb_train',
                    type=str,
                    help='The path of the data directory')
parser.add_argument('--val_dir',
                    default='../data/lmdb_val',
                    type=str,
                    help='The path of the data directory')
parser.add_argument(
    '--test',
    type=str,
    default=None,
    help='Evaluate the model with the specified name. Default: None')
parser.add_argument('--imgH',
                    type=int,
                    default=32,
                    help='the height of the input image to network')
parser.add_argument('--imgW',
                    type=int,
                    default=100,
                    help='the width of the input image to network')
parser.add_argument('--num_units',
                    type=int,
                    default=256,
                    help='size of the lstm hidden state')
parser.add_argument('--alphabet',
                    type=str,
                    default='0123456789abcdefghijklmnopqrstuvwxyz')

parser.add_argument('--val_steps', type=int, default=500)
parser.add_argument('--logging_steps', type=int, default=100)
parser.add_argument('--saving_steps', type=int, default=500)

parser.add_argument('--ckpt_dir',
                    default='./results',
                    type=str,
                    help='The path of the checkpoint directory')
parser.add_argument('--log_dir', default='./runs', type=str)

args = parser.parse_args()


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


if __name__ == '__main__':
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.train_dir):
        os.mkdir(args.train_dir)

    if not args.test:

        model = CRNN(num_channels=1,
                     num_class=(len(args.alphabet) + 1),
                     num_units=args.num_units)
        model.apply(weights_init)
        model.to(device)

        optimizer = optim.Adadelta(model.parameters(), lr=args.learning_rate)
        criterion = torch.nn.CTCLoss()

        converter = srtLabelConverter(args.alphabet)
        dataset_train = lmdbDataset(root=args.data_dir,
                                    imgH=args.imgH,
                                    imgW=args.imgW)

        dataset_val = lmdbDataset(root=args.val_dir,
                                  imgH=args.imgH,
                                  imgW=args.imgW)

        dataloader = DataLoader(dataset=dataset_train,
                                batch_size=args.batch_size,
                                shuffle=True)
        dataloader_val = DataLoader(dataset=dataset_val,
                                    batch_size=args.batch_size,
                                    shuffle=False)
        tb_writer = SummaryWriter(args.log_dir)

        for epoch in range(args.num_epochs):
            i = 0
            log_time = 0
            epoch_start_time = time.time()

            for img, label in dataloader:
                model.train()
                start_time = time.perf_counter()
                model.zero_grad()
                target, target_length = converter.encode(label)
                img.to(device)
                target.to(device)
                target_length.to(device)

                preds = model(img)
                preds_length = torch.IntTensor(
                    [preds.size(0) * args.batch_size]).to(device)
                loss = criterion(preds, target, preds_length, target_length)
                loss.backward()
                optimizer.step()
                step_time = (time.perf_counter() - start_time) / 1e3
                log_time += step_time

                i += 1
                if (i + 1) % args.logging_steps == 0:
                    tb_writer.add_scalar("loss", loss.item(), global_step=i)
                    tb_writer.add_scalar("time", log_time, global_step=i)
                    log_time = 0
                if (i + 1) % args.val_steps == 0:
                    model.eval()

                if (i + 1) % args.saving_steps == 0:
                    os.makedirs(args.ckpt_dir, exist_ok=True)
                    path = os.path.join(args.ckpt_dir,
                                        "crnn{}_{}".format(epoch, i))
                    torch.save(model.state_dict(), path)
