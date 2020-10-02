import os
import torch
import torch.nn
from hparam import hps
import numpy as np
import json


class GenreData(torch.utils.data.Dataset):

    # ['train', 'val', 'test']
    def __init__(self, hps, x, y):
        super(GenreData, self).__init__()
        self.x = x
        self.y = y

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    def __len__(self, ):
        return self.x.shape[0]

def load_json_data():
    with open('genre.json', 'r') as f:
        genre_dict =  json.load(f)
    label2index = genre_dict
    index2label = {v:k for k, v in genre_dict.items()}
    return label2index, index2label


def load_dataset(hps, label2index):
    print('Load dataset...')
    data_dict = {}
    for set_name in os.listdir(hps.feature_path):
        data_arr, label_arr = [], []
        set_path = os.path.join(hps.feature_path, set_name)
        for label_file in os.listdir(set_path):
            label = label_file.split('.')[0]
            data = np.load(os.path.join(set_path, label_file))
            data_arr.append(data)
            label_arr += [label] * data.shape[0]
        stacked_data = np.vstack(data_arr)
        stacked_label = np.array([label2index[item] for item in label_arr])
        print('The shape of %s set is: data->%s, label->%s' % (set_name, stacked_data.shape, stacked_label.shape))
        data_dict[set_name] = (stacked_data, stacked_label)
    print('Dataset loaded.')
    return data_dict

def generate_data(hps):
    # load label information and source dataset (numpy data)
    label2index, index2label = load_json_data()
    data_dict = load_dataset(hps, label2index)
    # transform: numpy data --> torch dataset 
    train_data_set = GenreData(hps, data_dict['train'][0], data_dict['train'][1])
    test_data_set = GenreData(hps, data_dict['test'][0], data_dict['test'][1])
    val_data_set = GenreData(hps, data_dict['val'][0], data_dict['val'][1])
    
    train_loader = torch.utils.data.DataLoader(train_data_set, batch_size=hps.batch_size, 
    shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_data_set, batch_size=hps.batch_size, 
    shuffle=False, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_data_set, batch_size=hps.batch_size, 
    shuffle=True, num_workers=0)                                            
    
    return train_loader, test_loader, val_loader
