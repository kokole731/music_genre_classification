'''
model_archive.py

A file that contains neural network models.
You can also implement your own model here.
'''
import os
import sys
sys.path.append(os.path.abspath(''))

import torch
torch.manual_seed(123)
import torch.nn as nn
from hparam import hps

class Baseline(nn.Module):
    def __init__(self, hparams):
        super(Baseline, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv1d(hparams.num_mels, 32, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(8, stride=8)
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(8, stride=8)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4, stride=4))

        self.linear = nn.Sequential(
            nn.Linear(192, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, len(hparams.genres)))

        self.apply(self._init_weights)


    def forward(self, x):
        x = x.transpose(1, 2)   # 32, 128, 1024
        x = self.conv0(x)       # 32, 32, 127
        x = self.conv1(x)       # 32, 32, 15
        x = self.conv2(x)       # 32, 64, 3
        x = x.view(x.size(0), x.size(1)*x.size(2))  # 32, 192
        x = self.linear(x)
        return x

    def _init_weights(self, layer) -> None:
        if isinstance(layer, nn.Conv1d):
            nn.init.kaiming_uniform_(layer.weight)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)

# model = Baseline(hps)
# input = torch.randn(32, 1024, 128)

# output = model(input)
# print(output)