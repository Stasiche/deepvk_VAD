import csv
import json
import torch


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


def restore_model(model):
    local_model_path = 'min_eval.pt'
    model.load_state_dict(torch.load(local_model_path, map_location=torch.device('cpu')))
