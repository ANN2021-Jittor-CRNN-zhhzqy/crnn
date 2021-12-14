import argparse
import os
import time

import numpy as np
import jittor as jt

from tensorboardX import SummaryWriter

from model import CRNN, weights_init, show_weights
from adadelta import Adadelta
from dataset import lmdbDataset
from utils import strLabelConverter

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=int, help='added to ckpt and log filename')
parser.add_argument('--num_epochs', type=int, default=10,
                    help='Number of training epoch. Default: 10')
parser.add_argument('--batch_size', type=int, default=256,
                    help='The number of batch_size.')
parser.add_argument('--learning_rate', type=float, default=1e-3,
                    help='Learning rate during optimization. Default: 1e-3')
parser.add_argument('--data_dir', default='../data/lmdb_train1', type=str,
                    help='The path of the data directory')
parser.add_argument('--val_dir', default='../data/lmdb_val1', type=str,
                    help='The path of the data directory')
parser.add_argument('--test', type=str, default=None,
                    help='Evaluate the model with the specified name. Default: None')
parser.add_argument('--imgH', type=int, default=32,
                    help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=100,
                    help='the width of the input image to network')
parser.add_argument('--num_units', type=int, default=256,
                    help='size of the lstm hidden state')
parser.add_argument('--alphabet', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz')
parser.add_argument('--val_steps', type=int, default=200)
parser.add_argument('--logging_steps', type=int, default=50)
parser.add_argument('--saving_steps', type=int, default=200)
parser.add_argument('--ckpt_dir', type=str, default='./result',
                    help='The path of the checkpoint directory')
parser.add_argument('--log_dir', type=str, default='./run')

args = parser.parse_args()


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def fast_eval(model, criterion, dataloader, converter, batch_size, max_iter=100):
    model.eval()
    iterator = iter(cycle(dataloader))
    max_iter = min(max_iter, len(dataloader))
    losses = []
    count = 0
    for _ in range(max_iter):
        imgs, labels = next(iterator)
        target, target_length = converter.encode(labels)
        preds = model(imgs)  # (24, 256, 38)
        preds_length = jt.full((preds.size(1), ), val=preds.size(0))
        loss = criterion(preds, target, preds_length, target_length)
        losses.append(loss.item())

        preds = preds.permute((1, 0, 2))  # (256, 24, 38)
        encoded_texts = preds.argmax(dim=-1)[0]  # (256, 24)
        str_preds = converter.decode(encoded_texts)  # [str](256)
        for str_pred, label in zip(str_preds, labels):
            if str_pred == label.lower():
                count += 1

    return np.mean(losses), count / (max_iter * batch_size), str_preds, labels


if __name__ == '__main__':
    jt.flags.use_cuda = 0
    print(args)
    if args.version is None:
        print("must enter a version. e.g. 'python main.py --version 01'")
        exit(-1)
    print("version ", args.version)
    print("data_dir ", args.data_dir)
    print("val_dir ", args.val_dir)

    config = "ep{}_bs{}_v{}".format(args.num_epochs, args.batch_size, args.version)
    args.ckpt_dir = os.path.join(args.ckpt_dir, config)
    args.log_dir = os.path.join(args.log_dir, config)
    print("ckpt_dir ", args.ckpt_dir)
    print("log_dir ", args.log_dir)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    if not args.test:
        model = CRNN(num_channels=1, num_class=len(args.alphabet)+1, num_units=args.num_units, num_layers=2)
        model.apply(weights_init)

        # optimizer = Adadelta(model.parameters(), lr=args.learning_rate)
        optimizer = jt.nn.SGD(model.parameters(), lr=args.learning_rate)
        criterion = jt.CTCLoss(blank=0)

        converter = strLabelConverter(args.alphabet, ignore_case=True)
        dataset_train = lmdbDataset(root=args.data_dir, imgH=args.imgH, imgW=args.imgW)
        dataset_val = lmdbDataset(root=args.val_dir, imgH=args.imgH, imgW=args.imgW)
        dataloader = dataset_train.set_attrs(batch_size=args.batch_size, shuffle=True)
        dataloader_val = dataset_val.set_attrs(batch_size=args.batch_size, shuffle=False)
        tb_writer = SummaryWriter(args.log_dir)

        i = 0
        for epoch in range(args.num_epochs):
            log_time = 0
            epoch_start_time = time.time()
            print("Epoch %d Start..." % (epoch))

            losses = []
            ep_i = 0
            for img, label in dataloader:  # var[256,1,32,100], [str](256)
                """
                * lmdb_train: 7,224,586 / 256 = 28221 + 1 steps
                * lmdb_train1: ?
                """
                model.train()
                start_time = time.perf_counter()

                target, target_length = converter.encode(label)  # var[256,len], var[256]
                
                print("ep_i %d start..." % ep_i)
                print("img[last] ", img[255, 0, 31, 99])
                model.apply(show_weights)
                preds = model(img)
                print("ep_i %d end" % ep_i)
                preds_length = jt.full((preds.size(1), ), val=preds.size(0), dtype=jt.int)
                
                loss = criterion(preds, target, preds_length, target_length)
                print("loss ", loss)
                optimizer.backward(loss)
                losses.append(loss.item())
                print(losses)
                print("")

                step_time = (time.perf_counter() - start_time) / 1e3
                log_time += step_time

                if (ep_i + 1) % args.logging_steps == 0:
                    print("  epoch %d - %d, cycle %d, loss %.2f, time %.2f s" % (epoch, ep_i, i, np.mean(losses), log_time))
                    tb_writer.add_scaler("train loss", np.mean(losses), global_step=i)
                    tb_writer.add_scaler("time", log_time, global_step=i)
                    losses = []
                    log_time = 0

                if (ep_i + 1) % args.val_steps == 0:
                    val_loss, accuracy, result, groundtrue = fast_eval(
                        model, criterion, dataloader_val, converter, args.batch_size)
                    tb_writer.add_scalar("val loss", val_loss, global_step=i)
                    tb_writer.add_scalar("accuracy", accuracy, global_step=i)
                    tb_writer.add_text("target", groundtrue[0], global_step=i)
                    tb_writer.add_text("preds", result[0], global_step=i)

                if (ep_i + 1) % args.saving_steps == 0:
                    path = os.path.join(args.ckpt_dir, "e%d_i%d" % (epoch, ep_i))
                    model.save(path)

                i += 1
                ep_i += 1

            epoch_time = time.time() - epoch_start_time
            tb_writer.add_scaler("epoch time", epoch_time, global_step=epoch)
            print("Epoch %d Finish, epoch time %.2fs" % (epoch, epoch_time))
            print("")  # \n

    print("Testing...")

    print("Test Finish")
