import os
import torch
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional

from pre_processing.dataset_split import data_set_split
from hparam import hps
from dataset import generate_data
from model.baseline import Baseline

train_loader, test_loader, val_loader = generate_data(hps)

device = torch.device('cuda: 0') if hps.device == 1 else torch.device('cpu')
net = Baseline(hps)
learning_rate, num_epochs, momentum = hps.learning_rate, hps.num_epochs, hps.momentum
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(num_epochs):

    total_loss = 0.
    for idx, (data, label) in enumerate(train_loader):

        data = data.to(device)
        label = label.to(device)

        output = net(data)
        loss = loss_fn(output, label)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 5 == 1:
            print('epoch: %d, step: %d, loss: %.2f' % (epoch, idx, total_loss / (idx+1)))
    
            num_correct = 0
            for idx, (data, label) in enumerate(test_loader):
                
                data = data.to(device)
                label = label.to(device)
                
                output = net(data)
                predict_y = torch.max(output, dim=1)[1]
                num_correct += (predict_y == label).sum().item()
            
            acc = num_correct / len(test_loader.dataset)
            print(acc)
