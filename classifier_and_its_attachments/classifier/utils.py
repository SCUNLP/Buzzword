import numpy as np
import json

import pandas as pd

from Encoder import Encoder
import torch
from data_cheating_filtered import get_data
from process_data import remove_chars


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return 'cpu'


def train_embedding():
    with open('./filtered.json', 'r', encoding='utf-8') as f:
        examples = json.load(f)

    encoder = Encoder()
    encoder.to(try_gpu())
    labels = []

    embeddings = None

    for i in range(16, len(examples), 16):
        sentences = []
        words = []
        for idx in range(i - 16, i):
            if examples[idx]['drop'] == 1:
                continue
            sentence = remove_chars(examples[idx]['sentence'])
            sentences.append(sentence)
            words.append(examples[idx]['word'])

            if examples[idx]['quality'] == "bad":
                labels.append(0)
            else:
                labels.append(1)

        with torch.no_grad():
            part_embedding = encoder.encode(sentences, words)
        print(part_embedding.device)
        print(part_embedding.shape)

        if embeddings is None:
            embeddings = part_embedding
        else:
            embeddings = torch.cat((embeddings, part_embedding), dim=0)

        print(embeddings.shape)

    labels = np.array(labels)
    np.save("data/label_new.npy", labels)
    print(embeddings.shape)
    embeddings = embeddings.cpu().numpy()
    np.save('data/embeddings_new.npy', embeddings)


def filter_false():
    with open('./filtered.json', 'r', encoding='utf-8') as f:
        examples = json.load(f)
    for i in range(len(examples)):
        sentence = examples[i]['sentence']
        drop = examples[i]['drop']
        if drop == 1:
            print(sentence)
            check = int(input())
            if check == 0:
                examples[i]['drop'] = 0

    with open('./filtered.json', 'w', encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=4)


def arrange_buzzword_examples():
    raw = get_data()
    examples = []
    for key, value in raw.items():
        word = key
        sentences = value['examples']
        for sentence in sentences:
            examples.append({'word': word, 'sentence': sentence})
    with open('./buzzword.json', 'w', encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=4)


def save_csv(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    words = []
    examples = []

    for item in data:
        words.append(item['word'])
        examples.append(item['sentence'])

    res = pd.DataFrame({'word': words, 'example': examples})
    res.to_csv('buzzword_examples.csv', index=False, encoding='utf-8')


def buzzword_embeddings(examples, words_raw):
    encoder = Encoder()
    encoder.to(try_gpu())
    embeddings = None
    sentences = []
    words = []
    for i in range(len(examples)):
        sentence = examples[i]
        word = words_raw[i]
        sentence = remove_chars(sentence)
        word = remove_chars(word)

        pos = sentence.find(word)
        if len(sentence) < 508:
            pass
        elif pos + 254 > len(sentence):
            sentence = sentence[len(sentence) - 508:]
        elif pos - 254 < 0:
            sentence = sentence[:508]
        else:
            sentence = sentence[pos-254:pos+254]

        if word not in sentence:
            sentence = sentence + ', ' + word

        sentences.append(sentence)
        words.append(word)

    with torch.no_grad():
        embeddings = encoder.encode(sentences, words)

    return embeddings


def select_best():
    way1 = pd.read_csv('res/MLPout_1536.csv', encoding='utf-8')
    way2 = pd.read_csv('res/MLPout.csv', encoding='utf-8')

    # 按word分组，选择每个word中label最高的10个example
    df_top = way1.groupby('word', group_keys=False).apply(lambda x: x.nlargest(10, 'label'))
    df_top.to_csv('top10_MLPout_1536.csv', index=False, encoding='utf-8')

    df_top = way2.groupby('word', group_keys=False).apply(lambda x: x.nlargest(10, 'label'))
    df_top.to_csv('top10_MLPout.csv', index=False, encoding='utf-8')

    print(df_top)




def selected():
    df_selected = pd.read_csv('res/top10_MLPout.csv', encoding='utf-8')

    groups = df_selected.groupby('word')

    result = {}
    for word, group in groups:
        examples = group['example'].tolist()
        result[word] = {'example': examples}

    json_data = json.dumps(result, ensure_ascii=False, indent=4)
    print(json_data)

    with open('data_selected.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    selected()
