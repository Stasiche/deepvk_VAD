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

import csv


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


def evaluate(model, dataloader: DataLoader, threshold: float) -> Dict[str, float]:
    model.eval()
    device = model.device
    cnts = defaultdict(int)
    for batch, labels, lengths in tqdm(dataloader, desc='Evaluating...'):
        batch = batch.to(device)
        labels = labels.to(device)
        mask = length_to_mask(lengths).to(device)
        with torch.no_grad():
            out = model(batch)

            preds = (torch.softmax(out[mask], dim=1)[:, 1] > threshold).int().cpu().numpy()
            labels = labels[mask].cpu().numpy()

            update_cnts(preds, labels, cnts)

    return {key: val / cnts['frames_cnt'] for key, val in cnts.items() if key != 'frames_cnt'}


def restore_model(model, run_name, model_name):
    local_model_path = join('models', run_name, model_name)

    w_path = wandb.restore(f'models/{model_name}', f'stasiche/Deepvk_VAD/{run_name}', root=dirname(local_model_path))
    model.load_state_dict(torch.load(w_path.name, map_location=torch.device('cpu')))


run_name = '2b4pjq8n'
device = torch.device('cuda')

api = wandb.Api()
run = api.run(f'stasiche/Deepvk_VAD/{run_name}')
config = run.config
model = ModelGRU(config['n_rnn_layers'], rnn_dim=config['n_mels'] // 2)
model.to(device)
restore_model(model, run_name, 'min_eval.pt')
eval_dataset = LibriSpeechActivityDataset('../librispeech_dataset_prepare/activity_val.csv',
                                          '../librispeech_dataset', config['n_mels'])
eval_dataloader = DataLoader(eval_dataset, batch_size=100, collate_fn=collate_fn, shuffle=False)

with open('metrics_librispeech_gru.csv', 'w') as f:
    writer = csv.writer(f)
    for i, th in enumerate([0.4062, 0.625, 0.95]):
        res = evaluate(model, eval_dataloader, th)
        if i == 0:
            writer.writerow(['th'] + list(res.keys()))
        writer.writerow([th] + [round(el, 3) for el in res.values()])
