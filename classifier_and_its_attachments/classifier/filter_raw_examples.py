import numpy as np
import pandas as pd
import re

import torch

from Encoder import Encoder
from data import get_rawdata
from torch.utils.data import Dataset, DataLoader


def try_gpu(i=0):
    if torch.cuda.device_count() >= i+1:
        return torch.device(f'cuda:{i}')
    return 'cpu'


def remove_punctuation(text):
    punctuation_pattern = r'[^\w\s]'
    cleaned_text = re.sub(punctuation_pattern, '', text)

    return cleaned_text

class TextDataset(Dataset):
    def __init__(self, sentences, words):
        super(TextDataset, self).__init__()
        self.sentences = sentences
        self.words = words

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        word = self.words[idx]
        return sentence, word


rawdata = pd.read_csv('wb_all.csv')

for idx, row in rawdata.iterrows():
    rawdata.loc[idx, 'content'] = remove_punctuation(rawdata.loc[idx, 'content'])

words = get_rawdata().keys()

rawdata = rawdata['content']
encoder = Encoder()
encoder.to(try_gpu())
embeddings = None
keywords = []
retrieved = []

for word in words:
    for sentence in rawdata:
        if word in sentence:
            retrieved.append(sentence)
            keywords.append(word)
df = pd.DataFrame(retrieved)
df.to_csv('sentences.csv', encoding='utf-8', index=False)

