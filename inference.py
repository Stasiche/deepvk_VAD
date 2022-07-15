import os.path

import wandb

from src.utils import inference_collate_fn, restore_model
from src.fit_utils import length_to_mask
from src.dataset import LibriSpeechActivityInferenceDataset
from src.model import ModelGRU

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

th_to_label = {
    0.625: 'FA_1',
    0.1875: 'FR_1',
    0.4062: 'FA_FR',
}

run_name = '2b4pjq8n'
device = torch.device('cuda')

api = wandb.Api()
run = api.run(f'stasiche/Deepvk_VAD/{run_name}')
config = run.config
model = ModelGRU(config['n_rnn_layers'], rnn_dim=config['n_mels'] // 2)
model.to(device)
model.eval()
restore_model(model, run_name, 'min_eval.pt')
dataset = LibriSpeechActivityInferenceDataset('for_devs', config['n_mels'])
dataloader = DataLoader(dataset, batch_size=300, collate_fn=inference_collate_fn, shuffle=False)

os.makedirs('predicts', exist_ok=False)
for i, th in enumerate([0.625, 0.1875, 0.4062]):
    with open(os.path.join('predicts', f'predicts_{th_to_label[th]}.txt'), 'w') as f:
        for fns, batch, lengths, durations in tqdm(dataloader, desc='Predicting...'):
            batch = batch.to(device)
            mask = length_to_mask(lengths).to(device)
            with torch.no_grad():
                out = model(batch)
                preds = (torch.softmax(out[mask], dim=1)[:, 1] > th).int().cpu().tolist()

            cumlength = 0
            for i, (fn, duration) in enumerate(zip(fns, durations)):
                prev = cumlength
                cumlength += lengths[i].item()
                f.write(f'{os.path.basename(fn)}, {str(preds[prev:cumlength][:duration])}\n')

