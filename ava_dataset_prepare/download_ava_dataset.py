import asyncio
import aiohttp
import argparse
import os
from moviepy.editor import VideoFileClip


async def download_site(url, session, dir_path, name):
    async with session.get(url) as response:
        r = await response.read()
        fp = os.path.join(dir_path, name)
        with open(fp, 'wb') as f:
            f.write(r)
        VideoFileClip(fp).subclip(900, 1800).audio.write_audiofile(''.join(fp.split(".")[:-1])+'.wav', verbose=False, logger=None)
        os.remove(fp)


async def download_all_sites(dir_path):
    os.makedirs(dir_path, exist_ok=True)
    with open('ava_speech_file_names_v1.txt', 'r') as f:
        names_list = [el.strip() for el in f.readlines()]
    async with aiohttp.ClientSession() as session:
        tasks = []
        for name in names_list:
            task = asyncio.create_task(download_site(f'https://s3.amazonaws.com/ava-dataset/trainval/{name}', session, dir_path, name))
            tasks.append(task)
        return await asyncio.gather(*tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', default='ava_speech_dataset', help='save directory path')
    args = parser.parse_args()

    asyncio.run(download_all_sites(args.d))

