import glob
import yaml
import tqdm 
import pathlib
from contextlib import suppress

import librosa
import torch
import torchaudio


def convert(path: pathlib.Path, sample_rate:int, pattern:str):
    output_path = path.parent / (path.stem + '_converted')
    output_path.mkdir(parents=True, exist_ok=True)

    fnames = glob.glob(pattern, root_dir=path)
    for fname in tqdm.tqdm(fnames):
        audio, sr = librosa.load(path / fname, sr=sample_rate)
        try:
            torchaudio.save(output_path / fname, torch.from_numpy(audio[None]), sample_rate=sample_rate)
        except RuntimeError:
            new_folder = output_path / fname
            if new_folder.suffix:
                new_folder = new_folder.parent
            new_folder.mkdir(parents=True, exist_ok=True)

            torchaudio.save(output_path / fname, torch.from_numpy(audio[None]), sample_rate=sample_rate)

if __name__ == "__main__":
    try:
        with open('config.yml', 'r') as stream:
            config = yaml.safe_load(stream)
    except FileNotFoundError:
        with open('config.yaml', 'r') as stream:
            config = yaml.safe_load(stream)

    sample_rate = config['sample_rate']

    with suppress(Exception):
        path = pathlib.Path('data/UrbanSound8K/')
        convert(path, sample_rate=sample_rate, pattern="**/*.wav")

    path = pathlib.Path('data/ESC-50/audio/')
    convert(path, sample_rate=sample_rate, pattern="*.wav")

    path = pathlib.Path('data/RIR/')
    convert(path, sample_rate=sample_rate, pattern="*.wav")