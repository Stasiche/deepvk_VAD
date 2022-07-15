import csv
import os

import torch
from torch import nn
import torchaudio
from torchaudio.transforms import Spectrogram, MelSpectrogram, AmplitudeToDB, MFCC
from torchaudio.transforms import FrequencyMasking, TimeMasking
from torch.utils.data import Dataset, DataLoader

from src.utils import unpack_activities


class LibriSpeechActivityDataset(Dataset):
    def __init__(self, activities_file, dataset_dir, n_mels):
        self.transform = nn.Sequential(
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


class AvaSpeechActivityDataset(Dataset):
    def __init__(self, activities_file, dataset_dir, n_mels):
        self.transform = nn.Sequential(
            MelSpectrogram(sample_rate=16000, n_mels=n_mels, win_length=320, n_fft=1000),
        )

        self.activities = []
        with open(activities_file, 'r') as f:
            reader = csv.reader(f)
            for fn, acts in reader:
                self.activities.append((fn+'.wav', [int(act) for act in acts]))

        self.dataset_dir = dataset_dir

    def __len__(self):
        return len(self.activities)

    def __getitem__(self, idx):
        fn, labels = self.activities[idx]
        waveform, _ = torchaudio.load(os.path.join(self.dataset_dir, fn))

        features = self.transform(waveform).squeeze().transpose(0, 1)
        labels = torch.Tensor(labels).long()
        features = features[:labels.shape[0]]
        return features, labels, labels.shape[0]

