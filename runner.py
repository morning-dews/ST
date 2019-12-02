from __future__ import print_function
from models import *

from utils import Dictionary, get_args

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import json
import time
import random
import os
import pdb
import pickle as pkl

def package(data, volatile=False):
    """Package data for training / evaluation."""
    data = map(lambda x: json.loads(x), data)
    #data = list(data)
    data = sorted(data, key = lambda x: len(x['text']), reverse=True)
    dat = map(lambda x: map(lambda y: dictionary.word2idx.get(y, 0), x['text']), data)
    #dat = list(dat)
    maxlen = 0
    for item in dat:
        maxlen = max(maxlen, len(item))
    targets = map(lambda x: x['label'], data)
    lenth = map(lambda x: len(x['text']), data)
    #targets = list(targets)
    maxlen = min(maxlen, 500)
    for i in range(len(data)):
        if maxlen < len(dat[i]):
            dat[i] = dat[i][:maxlen]
        else:
            for j in range(maxlen - len(dat[i])):
                dat[i].append(dictionary.word2idx['<pad>'])
    dat = Variable(torch.LongTensor(dat), volatile=volatile)
    targets = Variable(torch.LongTensor(targets), volatile=volatile)
    lenth = Variable( torch.LongTensor(lenth), volatile = volatile )
    #pdb.set_trace()
    return dat.t(), targets, lenth


def evaluate(epoch_number):
    """evaluate the model while training"""
    model.eval()  # turn on the eval() switch to disable dropout
    total_loss = 0
    total_correct = 0
    total_spl = 0
    total_xrl = 0
    total_Xrl = 0
    total_Yrl = 0
    total_cl = 0
    total_ol = 0
    Ysave = []
    for batch, i in enumerate(range(0, len(data_val), args.batch_size)):
        data, targets, lenth = package(data_val[i:min(len(data_val), i+args.batch_size)], volatile=True)
        if args.cuda:
            data = data.cuda()
            targets = targets.cuda()
        hidden = model.init_hidden(data.size(1))
        x, y, x_re, X, Y, Y_fromX, X_fromY, pred, outp, outp_fromY = model.forward(data, hidden,lenth, "eval",epoch_number)
        Ysave.append( (Y.cpu(), pred.cpu(), targets.cpu()) )
        output_flat = pred.view(data.size(1), -1)
        loss, sparse_loss, x_re_loss, X_re_loss, Y_re_loss, class_loss, outp_loss= \
            criterion(x, y, x_re, X, Y, Y_fromX, X_fromY, pred, targets, data.size(1), outp, outp_fromY, lenth, epoch_number)
        total_loss += loss.data
        total_spl += sparse_loss.data
        total_xrl += x_re_loss.data
        total_Xrl += X_re_loss.data
        total_Yrl += Y_re_loss.data
        total_cl += class_loss.data
        total_ol += outp_loss.data

        prediction = torch.max(output_flat, 1)[1]
        total_correct += torch.sum((prediction == targets).float())

    ave_loss = total_loss / (len(data_val) // args.batch_size)
    ave_spl = total_spl / (len(data_val) // args.batch_size)
    ave_xrl = total_xrl / (len(data_val) // args.batch_size)
    ave_Xrl = total_Xrl / (len(data_val) // args.batch_size)
    ave_Yrl = total_Yrl / (len(data_val) // args.batch_size)
    ave_cl = total_cl / (len(data_val) // args.batch_size)
    ave_ol = total_ol / (len(data_val) // args.batch_size)

    if  epoch_number is 15:
        f = open("../Y.pkl","wb")
        pkl.dump(Ysave, f)
        f.close()
    return ave_loss, total_correct.data[0] / len(data_val), ave_spl, ave_xrl, ave_Xrl,ave_Yrl, ave_cl, ave_ol

def train(epoch_number, trnotev):
    global best_val_loss, best_acc
    model.train()
    total_loss = 0
    start_time = time.time()
    if trnotev:
        trmode = "train"
    else:
        trmode = "eval"
    for batch, i in enumerate(range(0, len(data_train), args.batch_size)):
        data, targets, lenth = package(data_train[i:i+args.batch_size], volatile=False)
        if args.cuda:
            data = data.cuda()
            targets = targets.cuda()
        #pdb.set_trace()
        hidden = model.init_hidden(data.size(1))
        x, y, x_re, X, Y, Y_fromX, X_fromY, pred, outp, outp_fromY = model.forward(data, hidden,lenth, trmode,epoch_number)
        loss, spl, xrel, Xrel, Yrel,cll,ol = criterion(x, y, x_re, X, Y, Y_fromX, X_fromY, pred, targets, data.size(1), outp, outp_fromY, lenth,epoch_number)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        total_loss += loss.data

        if batch % args.log_interval == 0 and batch > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.4f}'.format(
                  epoch_number, batch, len(data_train) // args.batch_size,
                  elapsed * 1000 / args.log_interval, total_loss[0] / args.log_interval))
            total_loss = 0
            start_time = time.time()

    evaluate_start_time = time.time()
    val_loss, acc, ave_spl, ave_xrl, ave_Xrl, ave_Yrl, ave_cl, ave_ol = evaluate(epoch_number)
    print('-' * 89)
    f = open("./log/"+name, 'a+')
    fmt = '| evaluation | time: {:5.2f}s | valid loss (pure) {:5.4f} | Acc {:8.4f} ' \
          '| sparse_loss {:5.4f} | x_re_loss {:5.4f} | X_re_loss {:5.4f} ' \
          '| Y_re_loss {:5.4f} | class_loss {:5.4f} | outp_loss {:5.4f}'
    print(fmt.format((time.time() - evaluate_start_time), val_loss, acc, ave_spl, ave_xrl, ave_Xrl, ave_Yrl, ave_cl, ave_ol))
    print(fmt.format((time.time() - evaluate_start_time), val_loss, acc, ave_spl, ave_xrl, ave_Xrl, ave_Yrl, ave_cl, ave_ol), file = f)
    f.close()

    print('-' * 89)
    if epoch_number is -16:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001
    # Save the model, if the validation loss is the best we've seen so far.
    if not best_val_loss or val_loss < best_val_loss:
        with open(args.save, 'wb') as f:
            torch.save(model, f)
        f.close()
        best_val_loss = val_loss
    else:  # if loss doesn't go down, divide the learning rate by 5.
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.2
    if not best_acc or acc > best_acc:
        with open(args.save[:-3]+'.best_acc.pt', 'wb') as f:
            torch.save(model, f)
        f.close()
        best_acc = acc
    with open(args.save[:-3]+'.epoch-{:02d}.pt'.format(epoch_number), 'wb') as f:
        torch.save(model, f)
    f.close()


if __name__ == '__main__':
    # parse the arguments
    args = get_args()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    # Load Dictionary
    assert os.path.exists(args.train_data)
    assert os.path.exists(args.val_data)
    print('Begin to load the dictionary.')
    dictionary = Dictionary(path=args.dictionary)

    best_val_loss = None
    best_acc = None

    vocabnum = len(dictionary)
    if args.cuda:
        c = torch.ones(1)
    else:
        c = torch.zeros(1)

    print("vocabnum:", vocabnum)
    model = Classifier({
        'dropout': args.dropout,
        'vocabnum': vocabnum,
        'layers': args.layers,
        'hiddensize': args.hiddensize,
        'embsize': args.emsize,
        'basenum': args.basenum,
        'basesize': args.basesize,
        'pooling': 'last',
        'nfc': args.nfc,
        'dictionary': dictionary,
        'word-vector': args.word_vector,
        'class-number': args.class_number,
        'pre':args.pre,
        'maxlenth':args.maxlenth
    }, c)
    if args.cuda:
        model = model.cuda()

    print(args)

    criterion = myLoss(0.5)
    if args.cuda:
        criterion = criterion.cuda()
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=[0.9, 0.999], eps=1e-8, weight_decay=0)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.01)
    else:
        raise Exception('For other optimizers, please add it yourself. '
                        'supported ones are: SGD and Adam.')
    print('Begin to load data.')
    data_train = open(args.train_data).readlines()
    data_val = open(args.val_data).readlines()
    global name
    name = str(args.seed)+str(args.pre)+'-'+str(args.name)+'-'+str(time.time())
    try:
        for epoch in range(args.epochs):
            print(args.trnotev)
            train(epoch, args.trnotev)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exit from training early.')
        data_val = open(args.test_data).readlines()
        evaluate_start_time = time.time()
        test_loss, acc = evaluate()
        print('-' * 89)
        fmt = '| test | time: {:5.2f}s | test loss (pure) {:5.4f} | Acc {:8.4f}'
        print(fmt.format((time.time() - evaluate_start_time), test_loss, acc))
        print('-' * 89)
        exit(0)
