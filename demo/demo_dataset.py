import os
import csv
import json

import torch
from torch import nn
import torchaudio
from torchaudio.transforms import MelSpectrogram
from torch.utils.data import Dataset


def unpack_activities(fn):
    activities = []
    with open(fn, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            activity = json.loads(row[1])
            res = []
            for act, diff in activity:
                for _ in range(diff):
                    res.append(act)
            activities.append((row[0], res))

    return activities


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
        waveform, _ = torchaudio.load(os.path.join(self.dataset_dir, fn + '.flac'))

        features = self.transform(waveform).squeeze().transpose(0, 1)
        labels = torch.Tensor(labels).long()
        if features.shape[0] != labels.shape[0]:
            features = features[:-1]
        return features, labels, waveform, fn
