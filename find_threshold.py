import wandb
import csv

from src.utils import collate_fn
from src.fit_utils import evaluate
from src.dataset import LibriSpeechActivityDataset
from src.model import ModelGRU

import torch
from torch.utils.data import DataLoader
from src.utils import restore_model


run_name = '2b4pjq8n'
# device = torch.device('cpu')
device = torch.device('cuda')

api = wandb.Api()
run = api.run(f'stasiche/Deepvk_VAD/{run_name}')
config = run.config
model = ModelGRU(config['n_rnn_layers'], rnn_dim=config['n_mels']//2)
model.to(device)
restore_model(model, run_name, 'min_eval.pt')

eval_dataset = LibriSpeechActivityDataset('activity_val.csv', 'dataset', train=False, n_mels=config['n_mels'])

eval_dataloader = DataLoader(eval_dataset, batch_size=500, collate_fn=collate_fn, shuffle=False)
cache = {}
l, r = 0, 1
for i in range(4):
    th = round((l + r) / 2, 4)
    fa, fr, _ = evaluate(model, eval_dataloader, th)
    fa, fr = round(fa, 3), round(fr, 3)
    cache[th] = (fa, fr)

    if fa == 0.01:
        break
    if fa > 0.01:
        l = th
    else:
        r = th
print(f'fa: {fa}, th: {th}')

l, r = 0, 1
for i in range(4):
    th = round((l + r) / 2, 4)
    if cache.get(th, False):
        fa, fr = cache[th]
    else:
        fa, fr, _ = evaluate(model, eval_dataloader, th)
    fa, fr = round(fa, 3), round(fr, 3)
    cache[th] = (fa, fr)

    if fr == 0.01:
        break
    if fr < 0.01:
        l = th
    else:
        r = th
print(f'fr: {fr}, th: {th}')

l, r = 0, 1
for i in range(5):
    th = round((l + r) / 2, 4)
    if cache.get(th, False):
        fa, fr = cache[th]
    else:
        fa, fr, _ = evaluate(model, eval_dataloader, th)
    fa, fr = round(fa, 3), round(fr, 3)
    cache[th] = (fa, fr)

    if fa == fr:
        break
    if fa > fr:
        l = th
    else:
        r = th
print(f'fa: {fa}, fr: {fr}, th: {th}')

with open(f'{run_name}_thresholds.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['th', 'fa', 'fr'])
    for k, v in cache.items():
        writer.writerow([k, *v])
