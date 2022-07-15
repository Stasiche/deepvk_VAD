import os.path

from datasets import LibriSpeechActivityDataset, AvaSpeechActivityDataset
import wandb

from src.utils import collate_fn, save_model
# from src.fit_utils import train_one_epoch, evaluate
from src.model import ModelGRU

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from typing import Tuple
from tqdm import tqdm
from os.path import join, dirname
from collections import defaultdict
from typing import Dict
import torch
import webrtcvad
import wandb
from io import BytesIO
import numpy as np

import matplotlib.pyplot as plt

import contextlib
import wave
from pydub import AudioSegment
import csv


def read_file(path):
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def length_to_mask(length):
    max_len = length.max().item()
    arranges = torch.arange(max_len, device=length.device, dtype=length.dtype).expand(len(length), max_len)
    mask = arranges < length.unsqueeze(1)
    return mask


def update_cnts(predictions, targets, cnts):
    cnts['frames_cnt'] += predictions.shape[0]
    cnts['fa_total'] += ((targets == 0) & (predictions == 1)).sum()
    cnts['fr_clean'] += ((targets == 1) & (predictions == 0)).sum()
    cnts['fr_noise'] += ((targets == 2) & (predictions == 0)).sum()
    cnts['fr_music'] += ((targets == 3) & (predictions == 0)).sum()
    cnts['no_voice_frames'] += (targets == 0).sum()
    cnts['voice_frames'] += (targets != 0).sum()


def evaluate(activities, aggressiveness):
    vad = webrtcvad.Vad(aggressiveness)
    cnts = defaultdict(int)
    for fp, labels in tqdm(activities, desc='Evaluating...'):
        data, sr = read_file(fp)
        preds = []
        for i in range(0, len(data) // 320):
            act = int(vad.is_speech(data[i * 320:i * 320 + 320], sr))
            preds.append(act)

        preds = np.array(preds[:len(labels)])
        labels = np.array(labels)

        update_cnts(preds, labels, cnts)

    return {key: val / cnts['frames_cnt'] for key, val in cnts.items() if key != 'frames_cnt'}


activities_file = '../ava_dataset_prepare/ava_activities.csv'
activities = []
with open(activities_file, 'r') as f:
    reader = csv.reader(f)
    for fn, acts in reader:
        activities.append((os.path.join('..', 'ava_speech_dataset_one_channel', fn + '.wav'),
                           [int(act) for act in acts]))


with open('metrics_avaspeech_webrtc.csv', 'w') as f:
    writer = csv.writer(f)
    for aggressiveness in range(0, 4):
        res = evaluate(activities, aggressiveness)
        if aggressiveness == 0:
            writer.writerow(['aggr'] + list(res.keys()))
        writer.writerow([aggressiveness] + [round(el, 3) for el in res.values()])
