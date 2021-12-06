import argparse
from genericpath import exists
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import os
import sys
import time

from tensorboardX import SummaryWriter

from model import CRNN, weights_init

sys.path.append('../')
from tools.dataset import lmdbDataset
from tools.utlis import strLabelConverter

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs',
                    type=int,
                    default=20,
                    help='Number of training epoch. Default: 20')
parser.add_argument('--batch_size',
                    type=int,
                    default=3,
                    help='The number of batch_size.')
parser.add_argument('--learning_rate',
                    type=float,
                    default=1e-3,
                    help='Learning rate during optimization. Default: 1e-3')
parser.add_argument('--data_dir',
                    default='../data/lmdb_train1',
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

parser.add_argument('--val_steps', type=int, default=5)
parser.add_argument('--logging_steps', type=int, default=1)
parser.add_argument('--saving_steps', type=int, default=5)

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


def fast_eval(model,
              criterion,
              dataloader,
              converter,
              device,
              batch_size,
              max_iter=100):

    model.eval()
    iterator = iter(cycle(dataloader))
    max_iter = min(max_iter, len(dataloader))
    losses = []
    count = 0
    for _ in range(max_iter):
        imgs, labels = next(iterator)
        target, target_length = converter.encode(labels)
        imgs.to(device)
        target.to(device)
        target_length = target_length.int()
        target_length.to(device)

        preds = model(imgs)
        preds_length = torch.full((preds.size(1), ),
                                  fill_value=int(preds.size(0)))
        loss = criterion(preds, target, preds_length, target_length)

        losses.append(loss.tolist())
        loss.item()

        _, preds = preds.max(dim=2)
        preds = preds.transpose(0, 1)
        str_pred = converter.decode(preds, preds_length, raw=False)
        for pred, label in zip(str_pred, labels):
            if pred == label.lower():
                count += 1
    return np.mean(losses), count / (max_iter * batch_size), str_pred, labels


if __name__ == '__main__':
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = "{}_{}".format(args.num_epochs, args.batch_size)
    args.ckpt_dir = os.path.join(args.ckpt_dir, config)
    args.log_dir = os.path.join(args.log_dir, config)

    os.makedirs(args.ckpt_dir, exist_ok=True)

    if not args.test:

        model = CRNN(device=device,
                     num_channels=1,
                     num_class=(len(args.alphabet) + 1),
                     num_units=args.num_units)
        model.apply(weights_init)
        model.to(device)

        optimizer = optim.Adadelta(model.parameters(), lr=args.learning_rate)
        criterion = nn.CTCLoss(blank=0)

        converter = strLabelConverter(args.alphabet)
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
        i = 0
        for epoch in range(args.num_epochs):

            log_time = 0
            epoch_start_time = time.time()
            losses = []
            for img, label in dataloader:
                model.train()
                start_time = time.perf_counter()
                model.zero_grad()
                target, target_length = converter.encode(label)

                img.to(device)
                target.to(device)
                target_length = target_length.int()
                target_length.to(device)

                preds = model(img)
                preds_length = torch.full((preds.size(1), ),
                                          fill_value=int(preds.size(0)))

                loss = criterion(preds, target, preds_length, target_length)
                loss.backward()
                losses.append(loss.tolist())
                loss.item()
                optimizer.step()
                step_time = (time.perf_counter() - start_time) / 1e3
                log_time += step_time

                i += 1
                if (i + 1) % args.logging_steps == 0:
                    tb_writer.add_scalar("train loss",
                                         np.mean(losses),
                                         global_step=i)
                    tb_writer.add_scalar("time", log_time, global_step=i)
                    print(np.mean(losses))
                    losses = []
                    log_time = 0

                if (i + 1) % args.val_steps == 0:
                    model.eval()
                    val_loss, accuracy, result, groundtrue = fast_eval(
                        model=model,
                        criterion=criterion,
                        dataloader=dataloader_val,
                        converter=converter,
                        device=device,
                        batch_size=args.batch_size)

                    tb_writer.add_scalar("val loss", val_loss, global_step=i)
                    tb_writer.add_scalar("accuracy", accuracy, global_step=i)
                    tb_writer.add_text("target", groundtrue[0], global_step=i)
                    tb_writer.add_text("preds", result[0], global_step=i)

                if (i + 1) % args.saving_steps == 0:
                    path = os.path.join(args.ckpt_dir,
                                        "crnn{}_{}".format(epoch, i))
                    torch.save(model.state_dict(), path)
            epoch_time = time.time() - epoch_start_time
            tb_writer.add_scalar("epoch time", epoch_start_time, global_step=epoch)
