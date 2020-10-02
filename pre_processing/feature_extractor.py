'''
Transformation:
raw data -> mel spectrogram -> npy file
'''
import os
import torch
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from hparam import hps

# resize the data_arr if the input is too long/short.
def resize_arr(arr, len = hps.duration):
    data_arr = np.zeros((len, arr.shape[1]))
    if arr.shape[0] >= len:
        data_arr = arr[: len]
    else:
        data_arr[:arr.shape[0]] = arr[:arr.shape[0]]
    return data_arr

def mel_spectrogram(file_name, hparam):
    y, sr = librosa.load(file_name, sr=hparam.sample_rate)
    S = librosa.stft(y, n_fft=hparam.fft_size, hop_length=hparam.hop_length, win_length=hparam.win_length)
    # D = librosa.amplitude_to_db(S, )
    # Use mel_filter to extract mel spectrogram
    mel_basis = librosa.filters.mel(hparam.sample_rate, n_fft=hparam.fft_size, n_mels=hparam.num_mels)
    mel_S = np.dot(mel_basis, abs(S))
    mel_S = np.log10(1+10*mel_S)
    return mel_S.T

def save_feature(hps):
    if not os.path.isdir(hps.feature_path):
        os.mkdir(hps.feature_path)
    set_list = ['train', 'val', 'test']
    for set_name in set_list:
        set_name = os.path.join(hps.feature_path, set_name)
        if not os.path.isdir(set_name):
            os.mkdir(set_name)
    root_dir = hps.dataset_target_path
    set_name = ""
    label_num = 0
    for root, dirs, files in os.walk(root_dir):
        # set_label_name = '/'.join(root.split('/')[-2:])
        set_name = root.split('/')[-2]
        label_name = root.split('/')[-1]
        label_num = len(files)
        if label_num <= 1:
            continue
        data_arr = []
        save_path = os.path.join(hps.feature_path, set_name, label_name+'.npy')
        if os.path.exists(save_path):
            continue
        for idx, file in enumerate(files):
            if not file.split('.')[-1] in ['mp3', 'wav']:
                continue
            mel_feature = mel_spectrogram(os.path.join(root, file), hps)

            # 时长不够当作异常点去除
            if mel_feature.shape[0] < hps.duration:
                continue
            mel_feature = resize_arr(mel_feature)
            data_arr.append(mel_feature)
            a = '*' * (int)((idx+1) / label_num * 50)
            b = '.' * (int)((1 - (idx+1) / label_num) * 50)

            print('\rSaving features in %s/%s:   %s-->%s   %d%%' % (set_name, label_name, a, b, (int)(idx/label_num*100)), end="")
        np.save(save_path, np.stack(data_arr))
        print()


