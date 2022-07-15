import csv
import json
from torch.nn.utils.rnn import pad_sequence
import torch
import wandb
import os
from os.path import join, dirname


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


def collate_fn(data):
    features, labels, lengths = zip(*data)
    features = pad_sequence(features, batch_first=True)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    lengths = torch.Tensor(lengths).int()
    return features, labels, lengths


def inference_collate_fn(data):
    fn, features, lengths, durations = zip(*data)
    features = pad_sequence(features, batch_first=True)
    lengths = torch.Tensor(lengths).int()
    return fn, features, lengths, durations


def save_model(model, name: str):
    model_path = os.path.join('models', f'{name}.pt')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    wandb.save(model_path)


def restore_model(model, run_name, model_name):
    local_model_path = join('models', run_name, model_name)

    w_path = wandb.restore(f'models/{model_name}', f'stasiche/Deepvk_VAD/{run_name}', root=dirname(local_model_path))
    model.load_state_dict(torch.load(w_path.name, map_location=torch.device('cpu')))