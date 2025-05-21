import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_classification
#from imblearn.under_sampling import RandomUnderSampler
from process_data import remove_chars



def apply_mask(sentences, words):
    for idx, (sentence, word) in enumerate(zip(sentences, words)):
        sentences[idx] = sentence.replace(word, '[MASK]')
    return sentences


def get_pos(sentences, mask=103):
    pos = []
    for sentence in sentences:
        for idx, token in enumerate(sentence):
            if sentence[idx] == mask:
                pos.append(idx)
                break
    return pos


def try_gpu(i=0):
    if torch.cuda.device_count() >= i+1:
        return torch.device(f'cuda:{i}')
    return 'cpu'


class OneDataset(Dataset):
    def __init__(self, data):
        super(OneDataset, self).__init__()
        self.words = []
        self.sentences = []
        for key, values in data.items():
            word = key
            sentences = values['examples']
            for sentence in sentences:
                if len(sentence) >= 512:
                    continue
                if word not in sentence:
                    continue
                self.words.append(remove_chars(word))
                self.sentences.append(remove_chars(sentence))


    def __len__(self):
        return len(self.words)
        # return len(self.X_resampled)

    def __getitem__(self, idx):
        return self.words[idx], self.sentences[idx]
        # return self.X_resampled.iloc[idx]['word'], self.X_resampled.iloc[idx]['sentence'], self.y_resampled.iloc[idx]


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('../Chinese_BERT')
        self.encoder = BertModel.from_pretrained('../Chinese_BERT')

    def encode(self, sentences, words):
        sentences = list(sentences)
        sentences = apply_mask(sentences, words)
        tokenized = self.tokenizer(text=sentences, padding=True)
        input_tokens = torch.tensor(tokenized['input_ids']).to(try_gpu())
        pos = get_pos(input_tokens)
        attention_mask = torch.tensor(tokenized['attention_mask']).to(try_gpu())
        embeddings = self.encoder(input_ids=input_tokens, attention_mask=attention_mask)
        embeddings = embeddings['last_hidden_state']
        cls_embeddings = embeddings[:, 0, :]
        masked_embeddings = embeddings[range(embeddings.shape[0]),  pos]
        return masked_embeddings

