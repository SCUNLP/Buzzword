from transformers import BertTokenizer, BertModel, BertForMaskedLM
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import os
import json

mask = 103


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def apply_mask(words, examples):
    pos = []
    idx = 0
    for word in words:
        length = len(word) - 2
        for i in range(len(examples[idx])):
            if not (examples[idx][i:i + length] - word[1:1 + length]).any():
                for j in range(length):
                    examples[idx+j][i+j:i+length] = torch.full(np.shape(word[1+j:1+length]), mask)
                pos.append(i)
                idx += length
                break
    return examples, pos


def padding_sequence(text):
    text = [torch.tensor(text[i]) for i in range(len(text))]
    text = pad_sequence(text, batch_first=True).to(try_gpu())
    return text


class CorpusDataset(Dataset):
    def __init__(self):
        super(CorpusDataset, self).__init__()
        corpus_path = './cleaned_examples.json'
        with open(corpus_path, 'r', encoding='utf-8') as f:
            self.corpus = json.load(f)

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        return  self.corpus[idx]


class BertSelecting():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('../Chinese_BERT/')
        self.model = BertForMaskedLM.from_pretrained('../Chinese_BERT/')
        self.model.to(try_gpu())

    def tokenize(self, text):
        return self.tokenizer(text, padding=True)

    def encode(self, words, examples):
        '''先完成tokenize和添加掩码'''
        words = model.tokenize(words)
        examples = model.tokenize(examples)
        words_tokens = torch.tensor(words['input_ids']).to(try_gpu())
        examples_tokens = torch.tensor(examples['input_ids']).to(try_gpu())
        example_mask = torch.tensor(examples['attention_mask']).to(try_gpu())
        example_type = torch.tensor(examples['token_type_ids']).to(try_gpu())
        examples_tokens, pos = apply_mask(words_tokens, examples_tokens)
        output = self.model(input_ids=examples_tokens, attention_mask=example_mask, token_type_ids=example_type)
        return output.logits, words_tokens, pos

    def decode(self, logits, words, pos):
        res = []
        pred = torch.softmax(logits, dim=-1)
        idx = 0

        for i, word in enumerate(words):
            prob = []
            word_length = len(word) - 2
            for j in range(word_length):
                prob.append(pred[idx+j][pos[i]+j][word[j+1]])
            prob = np.array(prob)
            res.append(float(prob[0]*prob[1]*prob[2]*prob[3]))
            idx += word_length
        return res


if __name__ == '__main__':
    model = BertSelecting()
    batch_size = 2

    with open('res_sequential.json', 'r', encoding='utf-8') as f:
        result = json.load(f)

    dataset = CorpusDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    for items in dataloader:
        words = items['word']
        examples = items['sentence']
        quality = items['quality']
        repeated_examples = [examples[i] for i in range(batch_size) for _ in range(len(words[i]))]

        with torch.no_grad():
           logits, words_tokens, pos = model.encode(words, repeated_examples)
           res = model.decode(logits, words_tokens, pos)
           for i in range(len(examples)):
               result.append({'example': examples[i], 'probability': res[i], 'quality': quality[i], 'word': words[i]})
           print(res)
           with open('res_sequential.json', 'w', encoding='utf-8') as f:
               json.dump(result, f, ensure_ascii=False, indent=4)

