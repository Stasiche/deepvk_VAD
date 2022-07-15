import os
from tqdm import tqdm

# 1. Сконвертируем транскрипции из LibriSpeech в формат, с которым работает mfa
os.makedirs('librispeech_dataset', exist_ok=True)
total = 0
for _ in os.walk('LibriSpeech'):
    total += 1

for path, dirs, files in tqdm(os.walk('LibriSpeech'), total=total):
    for fn in files:
        if not fn.endswith('.trans.txt'):
            continue

        with open(os.path.join(path, fn), 'r') as f:
            for line in f.readlines():
                line = line.strip('\n').strip().split()
                speech_fn, transcription = line[0], ' '.join(line[1:])

                with open(os.path.join('librispeech_dataset', speech_fn.split('.')[0]+'.lab'), 'w') as out:
                    out.write(transcription)

# 2. Проверим, что не осталось ничего лишнего
flacs, labs = set(), set()
for el in os.listdir('librispeech_dataset'):
    if el.endswith('.flac'):
        flacs.add(el.split('.')[0])
    elif el.endswith('.lab'):
        labs.add(el.split('.')[0])
    else:
        print(el)

print(len(flacs), len(labs), len(flacs)-len(labs))

# 3. Сгруппируем записи по спикерам
for fn in tqdm(os.listdir('librispeech_dataset')):
    if os.path.isfile(os.path.join('librispeech_dataset', fn)):
        speaker = fn.split('-')[0]
        os.makedirs(os.path.join('librispeech_dataset', speaker), exist_ok=True)
        os.rename(os.path.join('librispeech_dataset', fn), os.path.join('librispeech_dataset', speaker, fn))

# 4. MFA не падал при попытке обработать весь датасет, поэтому разделим на несколько кусочков
speakers_dirs = os.listdir('librispeech_dataset')
parts_num = 4
part_size = len(speakers_dirs) // parts_num + 1
for i in range(parts_num):
    os.makedirs(os.path.join('dataset_parted', f'part_{i+1}'), exist_ok=True)
    for speaker_name in speakers_dirs[i*part_size: (i+1)*part_size]:
        os.makedirs(os.path.join('dataset_parted', f'part_{i + 1}', speaker_name), exist_ok=True)
        for fn in os.listdir(os.path.join('librispeech_dataset', speaker_name)):
            os.rename(os.path.join('librispeech_dataset', speaker_name, fn),
                      os.path.join('dataset_parted', f'part_{i+1}', speaker_name, fn))

# 5. Проверим, что все файлы распределились
dataset_cnt = len(os.listdir('librispeech_dataset'))

dataset_parted_cnt = 0
for part in os.listdir('dataset_parted'):
    dataset_parted_cnt += len(os.listdir(os.path.join('dataset_parted', part)))

print(dataset_cnt, dataset_parted_cnt)

# 6. Следуем указаниям на https://montreal-forced-aligner.readthedocs.io/en/latest/first_steps/example.html#alignment-example
# чтобы получить выравнивания для датасета.

# mfa align ./dataset_parted/part_1 ./tabed_lexion.txt ./english.zip ./aligned_dataset -j 8 --clean
# mfa align ./dataset_parted/part_2 ./tabed_lexion.txt ./english.zip ./aligned_dataset -j 8 --clean
# mfa align ./dataset_parted/part_3 ./tabed_lexion.txt ./english.zip ./aligned_dataset -j 8 --clean
# mfa align ./dataset_parted/part_4 ./tabed_lexion.txt ./english.zip ./aligned_dataset -j 8 --clean

# 7. Собираем кусочки датасета в один большой
# for i in range(parts_num):
#     for speaker_name in os.listdir(os.path.join('dataset_parted', f'part_{i + 1}')):
#         for fn in os.listdir(os.path.join('dataset_parted', f'part_{i + 1}', speaker_name)):
#             os.rename(os.path.join('dataset_parted', f'part_{i + 1}', speaker_name, fn),
#                       os.path.join('librispeech_dataset', speaker_name, fn))
