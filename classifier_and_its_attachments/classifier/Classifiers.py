import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier_MLP(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, dropout_rate=0.5):
        super(Classifier_MLP, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )
        self.output_layer = nn.Linear(hidden_dim // 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.output_layer(x)
        return x.reshape(-1)


class Classifier_CNN(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, dropout=0.2):
        super(Classifier_CNN, self).__init__()
        self.layer1 = nn.Conv1d(nfeat, nhid1, kernel_size=3, padding=1)
        self.layer2 = nn.Conv1d(nhid1, nhid2, kernel_size=3, padding=1)
        self.layer3 = nn.Conv1d(nhid2, 1, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        return x

