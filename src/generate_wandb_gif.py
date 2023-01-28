import argparse
import shutil
from pathlib import Path

import wandb
from PIL import Image

"""
Adapted from: https://wandb.ai/_scott/gif-maker/reports/Create-Gifs-from-Images-Logged-to-W-B---VmlldzoyMTI4NDQx
"""

# camb-mphil/sensor-placement/24xmcwni


def images_to_gif(input_filenames, output_filename, duration):
    input_filenames.sort(key=lambda x: int(x.split('_')[-2])) #sort by step
    frames = [Image.open(image) for image in input_filenames]
    frame_one = frames[0]
    frame_one.save(f'{output_filename}.gif', format="GIF", append_images=frames,
                   save_all=True, duration=duration, loop=0)


def download_gif(run_path: str, plot_name: str, output_name: str=None, duration=100):
    """
    :param run_path: The run path of form <entity>/<project>/<run_id>
    :param plot_name: The name of the logged images (e.g. variance_plot_high)
    :param output_name: The name of the output file. gifs/<plot_name>.gif if None
    :param duration: The duration of each frame in the GIF
    """
    if output_name is None:
        Path("gifs").mkdir(parents=True, exist_ok=True)
        output_name = "gifs/" + plot_name
    api = wandb.Api()
    run = api.run(run_path)
    filenames = []
    for file in run.files():
        if file.name.startswith("media/images/" + plot_name) and file.name.endswith('.png'):
            file.download()
            filenames.append(file.name)

    try:
        images_to_gif(filenames, output_name, duration)
    finally:
        shutil.rmtree("media/images")







parser = argparse.ArgumentParser()
parser.add_argument('--run_path', type=str, required=True, help='The run path of form <entity>/<project>/<run_id>')
parser.add_argument('--plot_name', type=str, required=True, help='The name of the logged images (e.g. variance_plot_high)')
parser.add_argument('--duration', type=int, default=600, help='The duration of each frame in the GIF')
parser.add_argument('--output_name', type=str, default=None, help='The name of the output GIF (without \".gif\").'
                                                                  ' Default: gifs/<plot_name>.gif')

if __name__ == "__main__":
    args = parser.parse_args()
    download_gif(args.run_path, args.plot_name, duration=args.duration)
