import argparse
import random
import jittor as jt
import jittor.nn as nn

import numpy as np
from torch._C import _jit_try_infer_type
import torch.optim as optim

import numpy as np
import os
import sys
import time

from tensorboardX import SummaryWriter

from model import CRNN, weights_init
from dataset import lmdbDataset

from utils import strLabelConverter

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs',
                    type=int,
                    default=10,
                    help='Number of training epoch. Default: 10')
parser.add_argument('--batch_size',
                    type=int,
                    default=256,
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
                    default='../data/lmdb_val1',
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
                    default=32,
                    help='size of the lstm hidden state')
parser.add_argument('--alphabet',
                    type=str,
                    default='0123456789abcdefghijklmnopqrstuvwxyz')

parser.add_argument('--val_steps', type=int, default=5)
parser.add_argument('--logging_steps', type=int, default=5)
parser.add_argument('--saving_steps', type=int, default=20)

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

        preds = model(imgs)
        preds_length = jt.full((preds.size(1), ),
                               val=preds.size(0),
                               dtype=jt.int)

        loss = criterion(preds, target, preds_length, target_length)
        losses.append(loss.clone().item())
        
        encoded_texts = preds.argmax(dim=-1)[0].transpose(0, 1)  # (256, 24)

        str_preds = converter.decode(encoded_texts)  # [str](256)
        for str_pred, label in zip(str_preds, labels):
            if str_pred == label.lower():
                count += 1

    return np.mean(losses), count / (max_iter * batch_size),str_preds, labels


if __name__ == '__main__':
    print(args)
    jt.flags.use_cuda = jt.has_cuda
    # jt.flags.lazy_execution = 0

    config = "{}_{}".format(args.num_epochs, args.batch_size)
    args.ckpt_dir = os.path.join(args.ckpt_dir, config)
    args.log_dir = os.path.join(args.log_dir, config)

    os.makedirs(args.ckpt_dir, exist_ok=True)

    if not args.test:

        model = CRNN(num_channels=1,
                     num_class=(len(args.alphabet) + 1),
                     num_units=args.num_units)
        model.apply(weights_init)

        optimizer = nn.SGD(model.parameters(), lr=args.learning_rate)
        criterion = jt.CTCLoss(blank=0)

        converter = strLabelConverter(args.alphabet)
        dataset_train = lmdbDataset(root=args.data_dir,
                                    imgH=args.imgH,
                                    imgW=args.imgW)

        dataset_val = lmdbDataset(root=args.val_dir,
                                  imgH=args.imgH,
                                  imgW=args.imgW)

        dataloader = dataset_train.set_attrs(batch_size=args.batch_size,
                                             shuffle=True)
        dataloader_val = dataset_val.set_attrs(batch_size=args.batch_size,
                                               shuffle=False)
        tb_writer = SummaryWriter(args.log_dir)
        i = 0
        for epoch in range(args.num_epochs):

            log_time = 0
            ep_i = 0
            epoch_start_time = time.time()
            losses = []
            print("Epoch %d Start..." % (epoch))
            for img, label in dataloader:
                model.train()
                optimizer.zero_grad()

                start_time = time.perf_counter()
                target, target_length = converter.encode(label)

                preds = model(img)
                preds_length = jt.full((preds.size(1), ),
                                       val=preds.size(0),
                                       dtype=jt.int)

                loss = criterion(preds, target, preds_length, target_length)

                optimizer.backward(loss)
                optimizer.step()


                losses.append(loss.clone().item())
                
                step_time = (time.perf_counter() - start_time) / 1e3
                log_time += step_time

                if (ep_i + 1) % args.logging_steps == 0:
                    print("  epoch %d - %d, cycle %d, loss %.2f, time %.2f s" %
                          (epoch, ep_i, i, np.mean(losses), log_time))
                    tb_writer.add_scalar("train loss",
                                         np.mean(losses),
                                         global_step=i)
                    tb_writer.add_scalar("time", log_time, global_step=i)
                    losses = []
                    log_time = 0

                if (ep_i + 1) % args.val_steps == 0:
                    val_loss, accuracy, result, groundtrue = fast_eval(
                        model=model,
                        criterion=criterion,
                        dataloader=dataloader_val,
                        converter=converter,
                        batch_size=args.batch_size)
                    tb_writer.add_scalar("val loss", val_loss, global_step=i)
                    tb_writer.add_scalar("accuracy", accuracy, global_step=i)
                    tb_writer.add_text("target", groundtrue[0], global_step=i)
                    tb_writer.add_text("preds", result[0], global_step=i)

                if (ep_i + 1) % args.saving_steps == 0:
                    path = os.path.join(args.ckpt_dir,
                                        "e%d_i%d" % (epoch, ep_i))
                    model.save(path)

                i += 1
                ep_i += 1
            epoch_time = time.time() - epoch_start_time
            tb_writer.add_scalar("epoch time",
                                 epoch_start_time,
                                 global_step=epoch)
