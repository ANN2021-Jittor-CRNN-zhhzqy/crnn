import argparse
import os
import time

import numpy as np
import jittor as jt
import jittor.nn as nn

from tensorboardX import SummaryWriter
from Levenshtein import distance

from model import CRNN, weights_init
from adadelta import Adadelta
from dataset import lmdbDataset
from dataset_test import lmdbTestDataset
from utils import strLabelConverter, BKTree

parser = argparse.ArgumentParser()
parser.add_argument('--version',
                    type=int,
                    help='added to ckpt and log filename')
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
                    default='../data/lmdb_train',
                    type=str,
                    help='The path of the data directory')
parser.add_argument('--val_dir',
                    default='../data/lmdb_val',
                    type=str,
                    help='The path of the data directory')
parser.add_argument('--test_dir',
                    default='../data/lmdb_ic03_SceneTest',
                    type=str,
                    help='The path of the data directory')
parser.add_argument('--test',
                    type=str,
                    default=None,
                    help='Test model. Default: None')
parser.add_argument('--test_mode',
                    type=str,
                    default=None,
                    help='["svt", "iiit5k50", "iiit5k1000", "hunspell"]')
parser.add_argument('--threshold',
                    type=int,
                    default=3,
                    help='Evaluate specified model. Default: None')
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
parser.add_argument('--val_steps', type=int, default=2000)
parser.add_argument('--logging_steps', type=int, default=400)
parser.add_argument('--saving_steps', type=int, default=2000)
parser.add_argument('--ckpt_dir',
                    type=str,
                    default='./result',
                    help='The path of the checkpoint directory')
parser.add_argument('--log_dir', type=str, default='./run')

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
    max_iter = min(max_iter, len(dataloader) / batch_size)
    losses = []
    count = 0
    for _ in range(max_iter):
        imgs, labels = next(iterator)
        target, target_length = converter.encode(labels)
        preds = model(imgs)  # (24, 256, 38)
        preds_length = jt.full((preds.size(1), ),
                               val=preds.size(0),
                               dtype=jt.int)
        loss = criterion(preds, target, preds_length, target_length.clone())

        losses.append(loss.item())

        preds = preds.permute((1, 0, 2))  # (256, 24, 38)
        encoded_texts = preds.argmax(dim=-1)[0]  # (256, 24)
        str_preds = converter.decode(encoded_texts)  # [str](256)
        for str_pred, label in zip(str_preds, labels):
            if str_pred == label.lower():
                count += 1

    return np.mean(losses), count / (max_iter * batch_size), str_preds, labels


if __name__ == '__main__':
    jt.flags.use_cuda = jt.has_cuda
    jt.flags.lazy_execution = 0

    print(args)
    if args.version is None:
        if args.test is None:
            print("must enter a version if not test! e.g. 'python main.py --version 01'")
            exit(-1)
    print("version ", args.version)
    print("data_dir ", args.data_dir)
    print("val_dir ", args.val_dir)

    config = "ep{}_bs{}_v{}".format(args.num_epochs, args.batch_size,
                                    args.version)
    if args.test:
        config = "test_{}".format(args.test)
    args.ckpt_dir = os.path.join(args.ckpt_dir, config)
    args.log_dir = os.path.join(args.log_dir, config)

    if not args.test:
        print("ckpt_dir ", args.ckpt_dir)
        print("log_dir ", args.log_dir)
        os.makedirs(args.ckpt_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)

        model = CRNN(num_channels=1,
                     num_class=len(args.alphabet) + 1,
                     num_units=args.num_units)
        # model.apply(weights_init)
        model.load("result/ep10_bs256_v38/e9_i27999")
        optimizer = Adadelta(model.parameters(), lr=args.learning_rate)
        # optimizer = nn.SGD(model.parameters(), lr=args.learning_rate)
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

            print("Epoch %d Start..." % (epoch))

            losses = []
            ep_i = 0
            epoch_start_time = time.time()
            for img, label in dataloader:  # var[256,1,32,100], [str](256)
                """
                * lmdb_train: 7,224,586 / 256 = 28221 + 1 steps
                * lmdb_train1: ?
                """
                model.train()
                optimizer.zero_grad()

                start_time = time.perf_counter()
                # print("label[0] ", label[0])
                target, target_length = converter.encode(label)  # (256, maxlen); (256)
                # print("target[0] ", target[0].data)

                preds = model(img)  # (24, 256, 38)
                preds_length = jt.full((preds.size(1), ),
                                       val=preds.size(0),
                                       dtype=jt.int)  # (256)

                loss = criterion(preds, target, preds_length, target_length)

                optimizer.backward(loss)
                optimizer.step()

                losses.append(loss.clone().item())
                # print("  ep_i %d, loss %.4f" % (ep_i, loss.clone().item()))

                step_time = (time.perf_counter() - start_time) / 1e3
                log_time += step_time

                if (ep_i + 1) % args.logging_steps == 0:
                    # print("  epoch %d - %d, cycle %d, loss %.2f, time %.2f s" %
                    #       (epoch, ep_i, i, np.mean(losses), log_time))
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
                    print("epoch %d - %d/28221, val loss %.4f, val accu %.4f" %
                          (epoch, ep_i, val_loss, accuracy))
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
                # print("i = %d" % i)
                # print("")

            epoch_time = time.time() - epoch_start_time
            tb_writer.add_scalar("epoch time", epoch_time, global_step=epoch)
            print("Epoch %d Finish, epoch time %.2fs\n" % (epoch, epoch_time))
    else:
        print("Testing %s ..." % (args.test))
        print("log_dir ", args.log_dir)
        os.makedirs(args.log_dir, exist_ok=True)

        model = CRNN(num_channels=1,
                     num_class=len(args.alphabet) + 1,
                     num_units=args.num_units)
        model.load("result/ep10_bs256_v38/e4_i5999")  # 这里修改 load 的模型
        criterion = jt.CTCLoss(blank=0)
        criterion_test = jt.CTCLoss(blank=0, reduction='none')
        converter = strLabelConverter(args.alphabet)

        dataset_test = lmdbTestDataset(root=args.test_dir,
                                       imgH=args.imgH,
                                       imgW=args.imgW)
        dataloader_test = dataset_test.set_attrs(batch_size=args.batch_size,
                                                 shuffle=False)
        print("len(dataloader_test) ", len(dataloader_test))
        tb_writer = SummaryWriter(args.log_dir)

        lex_dict = {}
        if args.test_mode == "svt" or args.test_mode == "iiit5k50" or args.test_mode == "iiit5k1000" or args.test_mode == "hunspell":
            if args.test_mode == "svt":
                lex_path = "../data/svt_lex.txt"
            elif args.test_mode == "iiit5k50":
                lex_path = "../data/iiit5k_lex50.txt"
            elif args.test_mode == "iiit5k1000":
                lex_path = "../data/iiit5k_lex1000.txt"
            else:
                lex_path = "../data/Hunspell.txt"
            
            with open(lex_path, "r") as f:
                lines = f.read().split('\n')

            if args.test_mode == "hunspell":
                lex_dict["hunspell"] = [w for w in lines if len(w) >= 3]
            else:
                for line in lines:
                    name_list = line.split(' ')
                    if (len(name_list) < 2):
                        break
                    path_key = name_list[0]
                    lex_str = name_list[1]
                    lex_dict[path_key] = [w.lower() for w in lex_str.split(',')]

        losses = []
        test_start_time = time.time()
        count = 0
        for imgs, labels, paths in dataloader_test:
            """
            * lmdb_iiit5k_test: 3000 / 256 = 11 + 1 steps
            * lmdb_iiit5k_train: 2000 / 256 = 7 + 1 steps
            """
            model.eval()
            start_time = time.perf_counter()
            # print("label[0] ", label[0])
            target, target_length = converter.encode(labels)
            # print("target[0] ", target[0].data)

            preds = model(imgs)
            preds_length = jt.full((preds.size(1), ),
                                    val=preds.size(0),
                                    dtype=jt.int)

            loss = criterion(preds, target, preds_length, target_length)
            losses.append(loss.item())

            preds = preds.permute((1, 0, 2))  # (256, 24, 38)
            encoded_texts = preds.argmax(dim=-1)[0]  # (256, 24)
            str_preds = converter.decode(encoded_texts)  # [str](256)
            
            if args.test_mode is None:
                for str_pred, label in zip(str_preds, labels):
                    if str_pred == label.lower():
                        count += 1
            elif args.test_mode == "svt" or args.test_mode == "iiit5k50" or args.test_mode == "iiit5k1000" or args.test_mode == "hunspell":
                bkTree = None
                if args.test_mode == "hunspell":
                    bkTree = BKTree(lex_dict['hunspell'])
                    print("build bk tree!")
                for _str_pred, label, path_key, prob in zip(str_preds, labels, paths, preds):  # str, str, str, (24, 38)
                    str_pred = _str_pred
                    if args.test_mode == "hunspell":
                        lex = bkTree.find(str_pred, args.threshold)
                        # print("bk find for %s, len(lex): %d" % (str_pred, len(lex)))
                    else:
                        lex = [w for w in lex_dict[path_key] if distance(w, _str_pred) <= args.threshold]
                    #print(lex)
                    if len(lex) > 0:
                        encoded_lex, len_lex = converter.encode(lex)  # (len(lex), max_len), (len(lex))
                        loss = criterion_test(prob.repeat(len(lex), 1, 1).permute((1, 0, 2)), 
                                              encoded_lex, 
                                              jt.array([prob.size(0)] * len(lex)), 
                                              len_lex)
                        str_pred = lex[loss.argmin(dim=-1)[0].item()]
                        #print(str_pred)
                    if str_pred == label.lower():
                        count += 1

        test_time = time.time() - test_start_time
        test_accu = count / len(dataloader_test)
        test_loss = np.mean(losses)
        example_pred = str_pred
        example_label = label
        
        print("")
        print("test time %.4f\n" % test_time)
        print("test accu %.4f\n" % test_accu)
        print("test loss %.4f\n" % test_loss)
        print("test pred %s\n" % example_pred)
        print("test label %s\n" % example_label)

        tb_writer.add_scalar("test time", test_time)
        tb_writer.add_scalar("test accu", test_accu)
        tb_writer.add_scalar("test loss", test_loss)
        tb_writer.add_text("test pred", example_pred)
        tb_writer.add_text("test label", example_label)

        print("Test Finish")
