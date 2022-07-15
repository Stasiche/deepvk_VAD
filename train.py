import wandb

from src.utils import collate_fn, save_model
from src.fit_utils import train_one_epoch, evaluate
from src.dataset import LibriSpeechActivityDataset
from src.model import ModelGRU

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR

wandb.init(
    project='Deepvk_VAD',
    # mode='offline',
    config={
        'batch_size': 90,
        'grad_accum': 2,
        'epochs': 1000,
        'lr': 3e-4,
        'seed': 42,

        'model': 'ModelGRU',
        'w_decay': 1e-5,
        'n_rnn_layers': 1,
        'n_mels': 128,
        'clip_grad_p': 2
    }
)

config = wandb.config
torch.manual_seed(config.seed)
device = torch.device('cuda')

model = ModelGRU(config.n_rnn_layers, rnn_dim=config.n_mels//2)
model.to(device)
optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.w_decay)
scheduler = ExponentialLR(optimizer, gamma=0.99945)

train_dataset = LibriSpeechActivityDataset('../activity_train.csv', '../dataset', train=True, n_mels=config.n_mels)
eval_dataset = LibriSpeechActivityDataset('../activity_val.csv', '../dataset', train=False, n_mels=config.n_mels)

train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=collate_fn, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=3*config.batch_size, collate_fn=collate_fn, shuffle=False)
step = 0
min_valloss, min_fa, min_fr = 1, 1, 1
for epoch in range(config.epochs):
    step = train_one_epoch(model, train_dataloader, optimizer, scheduler, step, epoch, config.grad_accum)
    fa, fr, eval_loss = evaluate(model, eval_dataloader, 0.5)
    wandb.log({'fa': fa, 'fr': fr, 'eval_loss': eval_loss}, step=step)
    if min_valloss > eval_loss:
        save_model(model, 'min_eval')
        min_valloss = eval_loss
    if min_fa > fa:
        save_model(model, 'min_fa')
        min_fa = fa
    if min_fr > fr:
        save_model(model, 'min_fr')
        min_fr = fr

