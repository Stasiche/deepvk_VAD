import requests
import argparse
import os
from moviepy.editor import VideoFileClip


def download_site(url, session, dir_path, name):
    with session.get(url) as response:
        r = response.content
        fp = os.path.join(dir_path, name)
        with open(fp, 'wb') as f:
            f.write(r)
        VideoFileClip(fp).subclip(900, 1800).audio.write_audiofile(''.join(fp.split(".")[:-1])+'.wav', verbose=False, logger=None)
        os.remove(fp)


def download_all_sites(dir_path):
    os.makedirs(dir_path, exist_ok=True)
    with open('ava_speech_file_names_v1.txt', 'r') as f:
        names_list = [el.strip() for el in f.readlines()]
    with requests.Session() as session:
        for name in names_list:
            download_site(f'https://s3.amazonaws.com/ava-dataset/trainval/{name}', session, dir_path, name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', default='ava_speech_dataset', help='save directory path')
    args = parser.parse_args()

    download_all_sites(args.d)

