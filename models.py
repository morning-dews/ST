from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import os
import pdb
import math
import pickle as pkl


class LSTM(nn.Module):

    def __init__(self, config, embedding):
        super(LSTM, self).__init__()
        self.drop = nn.Dropout(config['dropout'])
        self.encoder = embedding
        self.bilstm = nn.LSTM(config['embsize'], config['hiddensize'], config['layers'], dropout=config['dropout'])
        self.nlayers = config['layers']
        self.nhid = config['hiddensize']
        self.pooling = config['pooling']
        self.dictionary = config['dictionary']
#        self.init_weights()
        self.encoder.weight.data[self.dictionary.word2idx['<pad>']] = 0

    def forward(self, inp, hidden, lenth, cuda_av):
        emb = self.drop(self.encoder(inp))
        pack = nn.utils.rnn.pack_padded_sequence(emb, lenth)
        packed, hc = self.bilstm(pack, hidden)
        unpacked = nn.utils.rnn.pad_packed_sequence(packed)
        outp_all = unpacked[0]
        lenth = unpacked[1]
        if self.pooling == 'mean':
            outp = torch.mean(outp_all, 0).squeeze()
        elif self.pooling == 'max':
            outp = torch.max(outp_all, 0)[0].squeeze()
        elif self.pooling == 'last':
            outp = hc[0][1]
        return outp, emb, outp_all

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))

class Classifier(nn.Module):
#hiddensize = basenum
    def __init__(self, config, cuda_av):#"dropout": the dropout rate
                               #"embsize": embedding size
                               #"vocabnum": number of words
                               #"basenum": the number of base vectors
                               #"basesize": base vector size
                               #"dictionary": dic of words and ids
                               #"hiddensize": the size of hidden vector of lstm
                               #"layers": the number of lstm layers
        super(Classifier, self).__init__()
        self.embsize = config["embsize"]
        self.basenum = config['basenum']
        self.basesize = config['basesize']
        self.dictionary = config['dictionary']
        self.embedd = nn.Embedding(config["vocabnum"], self.embsize)
        self.maxlenth = config['maxlenth']
        self.base = torch.nn.Parameter(torch.randn(self.basesize, self.basenum))

        self.encoder = LSTM(config, self.embedd)
        self.fc = nn.Linear(config['basenum'], config['nfc'])

        self.w1 = nn.Linear(self.embsize, self.basesize, bias=False)
        self.w2 = nn.Linear(self.basesize, self.embsize, bias=False)
        self.w3 = nn.Linear(self.embsize, self.embsize, bias=False)
        self.w4 = nn.Linear(self.embsize, self.embsize, bias=False)

        self.drop = nn.Dropout(config['dropout'])
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.pred = nn.Linear(config['nfc'], config['class-number'])
        self.pred2 = nn.Linear(512, config['class-number'])
        self.cuda_av = cuda_av
        self.pre = config['pre']

    def Myleaky(self, x):
        flag = torch.sign(x)
        temp = torch.max(0.5*x, -0.5*x)
        temp2 = torch.max(1.5*x-0.5, -1.5*x-0.5)
        temp = torch.max(temp, temp2)
        return flag*temp

    def Dnor(self,x):
        x=3*x
        y = 2.7**( -((x-2)**2) ) - 2.7**( -((-x-2)**2) )
        #y = self.tanh(x)
        return y 

    def den2sp(self, inpx):
        mid_x = self.w1(inpx)
        mid_x = mid_x.transpose(0, 1)
        size_t = mid_x.size()
        outy = torch.mm(mid_x.contiguous().view(-1,size_t[-1]), self.base).view(size_t[0],size_t[1],-1)
        outy = outy.transpose(0, 1)
        outy = self.Dnor(outy)  # sparse representation
        return outy

    def sp2den(self, inpy):
        size_t = inpy.size()
        mid_y = inpy.contiguous().view(-1,size_t[2])
        mid_y = mid_y.transpose(0, 1)
        outx = torch.mm(self.base, mid_y)
        outx = outx.transpose(0, 1)
        outx = outx.view(size_t[0],size_t[1],-1)
        #outx = outx.transpose(0, 2)
        outx = self.w2(outx)
        outx = self.tanh(outx)#regain dense word representation
        outx = self.w3(outx)
        outx = self.tanh(outx)
        return outx

    def select_index_from_length(self, lengths, max_len=None):
        """
        Creates a boolean mask from sequence lengths.
        """
        batch_size = lengths.size()[0]
        max_len = max_len or lengths.max()
        return (torch.arange(0, max_len)
                .type_as(lengths)
                .repeat(batch_size, 1)
                .eq(lengths.unsqueeze(1)-1))

    def forward(self, inp, hidden,lenth,model,epoch=0):

        if epoch>200:
            for i in self.w1.parameters():
                i.requires_grad = False
            for i in self.w2.parameters():
                i.requires_grad = False
            for i in self.w3.parameters():
                i.requires_grad = False
            for i in self.embedd.parameters():
                i.requires_grad = False
            self.base.requires_grad = False
        # predict the class
        outp, _, outp_all= self.encoder.forward(inp, hidden, lenth, self.cuda_av)
        #dense and sparse words
        x = self.embedd(inp)  #dense word representaition
	x = self.tanh(x)
        y = self.den2sp(x)
        #pdb.set_trace()
        x_re = self.sp2den(y)

        #dense and sparse sentense
        X = outp_all  #dense sentense
        t1 = y.size()
        yset = [ self.Myleaky(y[0].reshape(1,t1[1],t1[2]) ) ]
        for i in range(t1[0]-1):
            tempy = self.Myleaky( -self.relu(-y[i]) + self.relu(y[i+1]) )
            tempy = tempy.reshape(1,t1[1],t1[2])
            yset.append(tempy)
        yset = tuple(yset)
        Y = torch.cat(yset, 0)
        Y = torch.cumsum(Y, 0)  #sparse sentense
        Y_temp = self.relu(Y-1)+self.relu(-Y-1)+1
        Y = Y/Y_temp
        packY = nn.utils.rnn.pack_padded_sequence(Y, lenth)
        Y = nn.utils.rnn.pad_packed_sequence(packY)
        length = Y[1]
        Y = Y[0]


        Y_fromX = self.den2sp(X)
        X_fromY = self.sp2den(Y)

        if self.pre == "lstm":
            outp = outp.view(outp.size(0), -1)
            fc = self.tanh(self.fc(self.drop(outp)))
            pred = self.pred(self.drop(fc))

        else:
            select_mask = self.select_index_from_length(lenth).unsqueeze(2).repeat(1,1,5).repeat(1,1,5).repeat(1,1,5).repeat(1,1,4).repeat(1,1,4).cuda()
            # outp_fromY = X_fromY.index_select(0, lenth.cuda()-1).transpose(0,1)
            outpY = Y.transpose(0,1).masked_select(select_mask).view(-1, self.basenum)
            fc = outpY
            #outp_fromY = outp

            select_mask2 = self.select_index_from_length(lenth).unsqueeze(2).repeat(1,1,self.embsize).cuda()
            outp_fromY = X_fromY.transpose(0,1).masked_select(select_mask2).view(-1, self.embsize)
            

            fc = self.tanh(self.fc(self.drop(fc)))
            pred = self.pred(self.drop(fc))

        #pdb.set_trace()
        return x, y, x_re, X, Y, Y_fromX, X_fromY, pred, outp, outp_fromY

    def init_hidden(self, bsz):
        return self.encoder.init_hidden(bsz)

    def encode(self, inp, hidden):
        return self.encoder.forward(inp, hidden)[0]

    def init_emb_weights(self, init_range=0.1):
        self.encoder.weight.data.uniform_(-init_range, init_range)

    def init_weights(self, init_range=0.1):
        self.fc.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.fill_(0)
        self.pred.weight.data.uniform_(-init_range, init_range)
        self.pred.bias.data.fill_(0)
        self.w1.weight.data.uniform_(-init_range, init_range)
        self.w2.weight.data.uniform_(-init_range, init_range)
        self.w3.weight.data.uniform_(-init_range, init_range)
        self.w4.weight.data.uniform_(-init_range, init_range)
        self.embedd.weight.data.uniform_(-init_range, init_range)


class myLoss(nn.Module):
    def __init__(self, threshould):
        super(myLoss, self).__init__()
        self.threshold = threshould
        self.c_e = nn.CrossEntropyLoss()
    
    def mymean(self, x, y, lenth):
        dif = ( x-y )**2
        dif = torch.sum(dif,0)
        dif = torch.mean(dif,1)
        dif = dif/lenth
        dif = torch.mean(dif)
        return dif
    
    def forward(self, x, y, x_re, X, Y, Y_fromX, X_fromY, pred, targets, size, outp, outp_fromY, lenth, epoch):
        lenth = lenth.float().cuda()
        sparse_loss = self.mymean(torch.sin( 3 * y ), 0, lenth )
        x_re_loss = self.mymean(x, x_re, lenth )
        X_re_loss = self.mymean( X, X_fromY, lenth )/(0.5*(self.mymean(X, 0, lenth) + self.mymean(X_fromY, 0 ,lenth)))
        Y_re_loss = self.mymean(Y, 1.1*Y_fromX, lenth )
        class_loss = self.c_e(pred.view(size, -1), targets)
        outp_loss = torch.mean( (outp - outp_fromY)**2 ) / ( 0.5 * (torch.mean(outp**2) + torch.mean(outp_fromY**2)) )
        loss = sparse_loss + x_re_loss + 0.8*X_re_loss + 0.8*Y_re_loss + class_loss
        return loss, sparse_loss, x_re_loss, X_re_loss, Y_re_loss, class_loss, outp_loss


