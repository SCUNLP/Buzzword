import torch
import json
import numpy as np
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


from Encoder import OneDataset


def try_gpu(i=0):
    if torch.cuda.device_count() >= i+1:
        return torch.device(f'cuda:{i}')
    return 'cpu'


class OneDataset(Dataset):
    def __init__(self, embeddings, labels):
        super(OneDataset, self).__init__()
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def preprocess_data(data, scaler=None):
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)

    data = data.cpu()
    if scaler is None:
        scaler = StandardScaler()
        data_numpy = scaler.fit_transform(data.detach().numpy())
    else:
        data_numpy = scaler.transform(data.detach().numpy())

    data_tensor = torch.FloatTensor(data_numpy).to(try_gpu())

    dump(scaler, 'scaler.joblib')
    return data_tensor


def trainer(classifier, encoder, epoch=5, args=None):
    #加载训练数据
    embeddings = np.load('data/embeddings_new.npy')[:, -768:]
    buzz_embeddings = np.load('data/embeddings_buzzword.npy')[:, -768:]
    labels_new = np.load('data/label_new.npy')
    preprocess_data(np.concatenate((embeddings, buzz_embeddings), axis=0))
    embeddings = preprocess_data(embeddings, scaler=load('./scaler.joblib'))
    labels_new = torch.from_numpy(labels_new)

    #切割测试集
    division = int(len(embeddings) / 5)
    train_embeddings = embeddings[:-division]
    train_labels = labels_new[:-division]
    test_embeddings = embeddings[-division:]
    test_labels = labels_new[-division:]

    #训练部分
    dataloader = DataLoader(OneDataset(train_embeddings, train_labels), batch_size=128, shuffle=True)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-5)
    loss = nn.BCEWithLogitsLoss()
    cnt = 0
    for epo in range(epoch):
        for embedding, labels in dataloader:
            classifier.train()
            optimizer.zero_grad()
            res = classifier(embedding)

            loss_num = loss(res, labels.to(res.device, dtype=torch.float))
            loss_num.backward()
            optimizer.step()
            cnt += 128
            test(classifier, test_embeddings, test_labels)
        torch.save(classifier.state_dict(), args.model + '.pth')


def test(classifier, embeddings, labels, index_test=None):
    total = 0
    correct = 0
    res = classifier(embeddings)
    res = F.sigmoid(res)
    res = torch.abs(res - labels.to(res.device))
    correct += torch.sum(res <= 0.5)
    total += res.shape[0]
    print('----test----', total, correct, correct/total)


def check(prediction, labels, index_test=None):
    with open('filtered.json', 'r', encoding='utf-8') as f:
        examples = json.load(f)
    for i in range(labels.shape[0]):
        idx = index_test[i]
        if labels[i] == 0 and prediction[i] > 0.5:
            examples[idx]['drop'] = 1
        else:
            examples[idx]['drop'] = 0
    with open('filtered.json', 'w', encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=4)

