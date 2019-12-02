import json
import pandas as pd
import random
from tqdm import tqdm
from nltk import word_tokenize
from collections import Counter
from collections import defaultdict
from sklearn.model_selection import train_test_split
from io import open
import re

class Dictionary(object):
    """Simple vocabulary wrapper."""
    def __init__(self, path=''):
        self.word2idx = defaultdict(int)
        self.idx2word = {}
        self.idx = 0
        if path != '':  # load an external dictionary
            words = json.loads(open(path, 'r').readline())
            for item in words:
                self.add_word(item)

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

if __name__ == '__main__':
    dictionary = Dictionary()
    dictionary.add_word('<unk>')
    dictionary.add_word('<pad>')  # add padding word

    
    file = open('../sentiment_dataset/data/stsa.fine.train', 'r', encoding='latin1')
    dataset = []
    for i, line in tqdm(enumerate(file)):
        label = int(line[0])
        line = line[1:]
        # words = tokenizer(unicode(' '.join(line.split())))
        data = {
            'label': label,
            # 'text': map(lambda x: x.text.lower(), words)
            'text': line.split()
        }
        dataset.append(data)

    counter = Counter()
    fout = open('train.json', 'w')
    for data in dataset:
        fout.write(unicode(json.dumps(data)) + '\n')
        counter.update(data['text'])
    fout.close()

    file = open('../sentiment_dataset/data/stsa.fine.test', 'r', encoding='latin1')
    dataset = []
    for i, line in tqdm(enumerate(file)):
        label = int(line[0])
        line = line[1:]
        # words = tokenizer(unicode(' '.join(line.split())))
        data = {
            'label': label,
            # 'text': map(lambda x: x.text.lower(), words)
            'text': line.split()
        }
        dataset.append(data)

    fout = open('test.json', 'w')
    for data in dataset:
        fout.write(unicode(json.dumps(data)) + '\n')
    fout.close()

    stop = None
    with open('../stop.txt', 'r') as f:
        lines = f.readlines()
        stop = set([text.strip() for text in lines])

    fout = open('test_remove25.json', 'w')
    for data in dataset:
        check = False
        for text in data['text']:
            if text in stop and random.randint(0,3) < 1:
                check = True
                data['text'].remove(text)
        if check:
            fout.write(unicode(json.dumps(data)) + '\n')
    fout.close()

    fout = open('test_remove50.json', 'w')
    for data in dataset:
        check = False
        for text in data['text']:
            if text in stop and random.randint(0,3) < 2:
                check = True
                data['text'].remove(text)
        if check:
            fout.write(unicode(json.dumps(data)) + '\n')
    fout.close()

    fout = open('test_remove75.json', 'w')
    for data in dataset:
        check = False
        for text in data['text']:
            if text in stop and random.randint(0,3) < 3:
                check = True
                data['text'].remove(text)
        if check:
            fout.write(unicode(json.dumps(data)) + '\n')
    fout.close()

    fout = open('test_remove100.json', 'w')
    for data in dataset:
        check = False
        for text in data['text']:
            if text in stop and random.randint(0,3) < 4:
                check = True
                data['text'].remove(text)
        fout.write(unicode(json.dumps(data)) + '\n')
    fout.close()

    file = open('../sentiment_dataset/data/stsa.fine.dev', 'r', encoding='latin1')
    dataset = []
    for i, line in tqdm(enumerate(file)):
        label = int(line[0])
        line = line[1:]
        # words = tokenizer(unicode(' '.join(line.split())))
        data = {
            'label': label,
            # 'text': map(lambda x: x.text.lower(), words)
            'text': line.split()
        }
        dataset.append(data)

    fout = open('dev.json', 'w')
    for data in dataset:
        fout.write(unicode(json.dumps(data)) + '\n')
    fout.close()

    for word in [k for k, v in counter.most_common(30000)]:
        dictionary.add_word(word)

    with open('dict.json', 'w') as dout:  # save dictionary for fast next process
        dout.write(unicode(json.dumps(dictionary.idx2word.values()))+'\n')
        dout.close()
