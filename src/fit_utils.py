import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import wandb
from tqdm.auto import tqdm

from typing import Tuple


def length_to_mask(length):
    max_len = length.max().item()
    arranges = torch.arange(max_len, device=length.device, dtype=length.dtype).expand(len(length), max_len)
    mask = arranges < length.unsqueeze(1)
    return mask


def train_one_epoch(model, dataloader: DataLoader, optimizer: torch.optim, scheduler: torch.optim.lr_scheduler,
                    step: int, epoch: int, grad_accum: int) -> int:
    device = model.device
    criterion = CrossEntropyLoss()
    model.train()
    total_loss = 0
    for batch, labels, lengths in tqdm(dataloader, desc='Traning...'):
        step += 1
        batch = batch.to(device)
        labels = labels.to(device)
        out = model(batch)

        loss = criterion(out.transpose(1, 2), labels)
        loss /= grad_accum
        loss.backward()
        total_loss += loss.item()
        if not step % grad_accum:
            torch.nn.utils.clip_grad_norm_(model.parameters(), wandb.config.clip_grad_p)
            optimizer.step()
            optimizer.zero_grad()
            if optimizer.param_groups[0]['lr'] > 1e-6:
                scheduler.step()
            wandb.log({'epoch': epoch, 'loss': total_loss, 'lr': optimizer.param_groups[0]['lr']}, step=step)
            total_loss = 0

    return step


def evaluate(model, dataloader: DataLoader, threshold: float) -> Tuple[float, float, float]:
    model.eval()
    criterion = CrossEntropyLoss()
    device = model.device
    fa_cnt, fr_cnt, frames_cnt = 0, 0, 0
    total_loss = 0
    for batch, labels, lengths in tqdm(dataloader, desc='Evaluating...'):
        batch = batch.to(device)
        labels = labels.to(device)
        mask = length_to_mask(lengths).to(device)
        with torch.no_grad():
            out = model(batch)
            loss = criterion(out.transpose(1, 2), labels)
            total_loss += loss.item()

            preds = (torch.softmax(out[mask], dim=1)[:, 1] > threshold).int().cpu().numpy()
            labels = labels[mask].cpu().numpy()

            false_mask = (preds != labels)
            fa_cnt += (false_mask & (preds == 1)).sum()
            fr_cnt += (false_mask & (preds == 0)).sum()
            frames_cnt += preds.shape[0]

    return fa_cnt / frames_cnt, fr_cnt / frames_cnt, total_loss / len(dataloader)
