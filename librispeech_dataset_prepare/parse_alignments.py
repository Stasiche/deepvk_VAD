import os
from praatio import textgrid as tgio
import csv
import json
from tqdm import tqdm

total = 0
for _ in os.walk('aligned_dataset'):
    total += 1

with open('activity.csv', 'w') as f:
    writer = csv.writer(f)
    for g_path, _, files in tqdm(os.walk('aligned_dataset'), total=total):
        for fn in files:
            path = os.path.join(g_path, fn)
            intervals_lst = tgio.openTextgrid(path, True).tierDict['words'].entryList

            activity = []
            interval = intervals_lst[0]
            act, diff = int(interval.label != ''), int(100 * interval.end) - int(100 * interval.start)
            buffer = [act, diff]

            for interval in intervals_lst[1:]:
                act, diff = int(interval.label != ''), int(100 * interval.end) - int(100 * interval.start)
                if act == buffer[0]:
                    buffer[1] += diff
                else:
                    activity.append(buffer)
                    buffer = [act, diff]

            buffer[1] += 1
            activity.append(buffer)

            writer.writerow([os.path.basename(path).split('.')[0], json.dumps(activity)])
