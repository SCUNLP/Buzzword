import pandas as pd
import torch
import numpy as np
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from data_cheating_filtered import get_data
from process_data import remove_chars
from utils import buzzword_embeddings

from Encoder import OneDataset


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return 'cpu'


def preprocess_data(data, scaler=None):
    # 确保输入是 PyTorch 张量
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)

    data = data.cpu()
    # 如果没有提供 scaler，创建一个新的
    if scaler is None:
        scaler = StandardScaler()
        data_numpy = scaler.fit_transform(data.detach().numpy())
    else:
        data_numpy = scaler.transform(data.detach().numpy())

    # 将 numpy 数组转回 PyTorch 张量
    data_tensor = torch.FloatTensor(data_numpy).to(try_gpu())

    dump(scaler, 'scaler.joblib')
    return data_tensor


class PredictDataset(Dataset):
    def __init__(self, examples, words):
        super(PredictDataset, self).__init__()
        self.examples = examples
        self.words = words

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx], self.words[idx]


def predict(encoder, classifier, args):
    raw = pd.read_csv(args.data_path)
    examples = raw['example']
    words = raw['word']

    dataloader = DataLoader(PredictDataset(examples, words), batch_size=10, shuffle=False)
    labels = []
    for example, word in dataloader:
        with torch.no_grad():
            embeddings = buzzword_embeddings(example, word)
        embeddings = preprocess_data(embeddings)
        classifier.eval()
        with torch.no_grad():
            res = classifier(embeddings)
            res = F.sigmoid(res)
            labels = labels + list(res.cpu().detach())

    labels = np.array(labels).reshape(-1)
    sentences = raw['example'].tolist()
    words = raw['word'].tolist()
    output = pd.DataFrame({'word': words, 'example': sentences, 'label': labels})
    output.to_csv('./res/' + args.model + f'out.csv', index=False)
    print("results have been saved in", args.model + f'out.csv')

