import os

import torch
from torch import nn
import torchaudio
from torchaudio.transforms import Spectrogram, MelSpectrogram, AmplitudeToDB, MFCC
from torchaudio.transforms import FrequencyMasking, TimeMasking
from torch.utils.data import Dataset, DataLoader

from src.utils import unpack_activities


class LibriSpeechActivityDataset(Dataset):
    def __init__(self, activities_file, dataset_dir, train, n_mels):
        if train:
            self.transform = nn.Sequential(
                MelSpectrogram(sample_rate=16000, n_mels=n_mels, win_length=320, n_fft=1000),
                FrequencyMasking(freq_mask_param=30),
                TimeMasking(time_mask_param=70),
            )
        else:
            self.transform = nn.Sequential(
                # MelSpectrogram(sample_rate=16000, n_mels=n_mels, n_fft=320),
                MelSpectrogram(sample_rate=16000, n_mels=n_mels, win_length=320, n_fft=1000),
            )

        self.activities = unpack_activities(activities_file)
        self.dataset_dir = dataset_dir

    def __len__(self):
        return len(self.activities)

    def __getitem__(self, idx):
        fn, labels = self.activities[idx]
        waveform, _ = torchaudio.load(os.path.join(self.dataset_dir, fn.split('-')[0], fn+'.flac'))

        features = self.transform(waveform).squeeze().transpose(0, 1)
        labels = torch.Tensor(labels).long()
        if features.shape[0] != labels.shape[0]:
            features = features[:-1]
        return features, labels, labels.shape[0]


class LibriSpeechActivityInferenceDataset(Dataset):
    def __init__(self, dataset_dir, n_mels):
        self.transform = nn.Sequential(
            MelSpectrogram(sample_rate=16000, n_mels=n_mels, win_length=320, n_fft=1000)
        )

        self.dataset_dir = [os.path.join(dataset_dir, el) for el in os.listdir(dataset_dir)]

    def __len__(self):
        return len(self.dataset_dir)

    def __getitem__(self, idx):
        fn = self.dataset_dir[idx]
        waveform, _ = torchaudio.load(fn)

        features = self.transform(waveform).squeeze().transpose(0, 1)

        return fn, features, features.shape[0], int(waveform.shape[1]/160)
