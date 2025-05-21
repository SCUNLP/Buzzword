import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import argparse

from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from torch.nn import init
from torch.utils.data import Dataset, DataLoader
from Encoder import Encoder
from Classifiers import Classifier_CNN, Classifier_MLP
from train import trainer
from infer import predict
from sklearn.model_selection import train_test_split
import random

random.seed(27)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', default=True, help='Train or infer')
    parser.add_argument('--model', type=str, default='MLP', choices=['MLP', 'CNN'], help='Model name(MLP or CNN)')
    parser.add_argument('--data_path', type=str, default='./buzzword_examples.csv', help='Data path')
    return parser.parse_args()


def try_gpu(i=0):
    if torch.cuda.device_count() >= i+1:
        return torch.device(f'cuda:{i}')
    return 'cpu'


def initialize_weights_xavier(module):
    if isinstance(module, torch.nn.Linear):
        init.xavier_uniform_(module.weight)


if __name__ == '__main__':
    args = parse_args()
    encoder = Encoder()
    if args.train:
        if args.model == 'MLP':
            classifier = Classifier_MLP(768)
        elif args.model == 'CNN':
            classifier = Classifier_CNN(0,0,0,0)
        initialize_weights_xavier(classifier)
        encoder.to(try_gpu())
        classifier.to(try_gpu())
        trainer(classifier=classifier, encoder=encoder, epoch=1, args=args)

    else:
        if args.model == 'MLP':
            classifier = Classifier_MLP(input_dim=768, dropout_rate=0)
            classifier.load_state_dict(torch.load('./res/' + args.model + '.pth'))
        elif args.model == 'CNN':
            classifier = Classifier_CNN(0,0,0,0)

        encoder.to(try_gpu())
        classifier.to(try_gpu())
        predict(encoder, classifier, args)

