import csv
from collections import defaultdict
import os
labels_to_num = {
    'NO_SPEECH': 0,
    'CLEAN_SPEECH': 1,
    'SPEECH_WITH_NOISE': 2,
    'SPEECH_WITH_MUSIC': 3,

}

activities = defaultdict(list)
check_sort = defaultdict(int)
with open('ava_speech_labels_v1.csv', 'r') as f:
    reader = csv.reader(f)
    for fn, start, end, label in reader:
        if not os.path.exists(f'../ava_speech_dataset_one_channel/{fn}.wav'):
            continue
        if check_sort.get(fn, False) and check_sort[fn] != start:
            raise RuntimeError('tabel is not sorted')
        check_sort[fn] = end
        activities[fn].append((round(float(end)-float(start), 2), labels_to_num[label]))

with open('ava_activities.csv', 'w') as out:
    writer = csv.writer(out)
    for fn, acts in activities.items():
        total_acts = []
        for duration, label in acts:
            for _ in range(int(duration*100)):
                total_acts.append(label)
        writer.writerow([fn, ''.join(map(str, total_acts))])
