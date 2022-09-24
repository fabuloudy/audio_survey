# Deep Learning Survey for Audio Data

## Abstract

...

## Datasets

#### ESC-50

[![kaggle](https://www.kaggle.com/static/images/favicon.ico)](https://www.kaggle.com/datasets/ludovick/esc50dataset) [![paperswithcode](https://paperswithcode.com/favicon.ico)](https://paperswithcode.com/dataset/esc-50) [![github](https://github.githubassets.com/favicons/favicon-dark.svg)](https://github.com/karolpiczak/ESC-50)

The ESC-50 dataset is a labeled collection of **2000** environmental audio recordings suitable for benchmarking methods of environmental sound classification.

The dataset consists of **5**-second-long recordings organized into **50** semantical classes (with **40** examples per class) loosely arranged into 5 major categories: `Animals`, `Natural soundscapes & water sounds`, `Human non-speech sounds`, `Interior/domestic sounds`, `Exterior/urban noises`

#### UrbanSound8K

[![kaggle](https://www.kaggle.com/static/images/favicon.ico)](https://www.kaggle.com/datasets/chrisfilo/urbansound8k) [![paperswithcode](https://paperswithcode.com/favicon.ico)](https://paperswithcode.com/dataset/urbansound8k-1)

This dataset contains **8732** labeled environmental sound excerpts (<=**4s**) of urban sounds from **10** classes: `air_conditioner`, `car_horn`, `children_playing`, `dog_bark`, `drilling`, `enginge_idling`, `gun_shot`, `jackhammer`, `siren` and `street_music`

## Setup 🔨

#### Config

sample_rate - project sasmple rate, default 44_100
mel_spectrogram - dict with argumetns for `torchaudio.transforms.MelSpectrogram` class. Default param: [1] 3.1. Feature Extraction  
* win_length - Window size in sec, calculated by sample_rate  
* hop_length - Length of hop between STFT windows in sec, calculated by sample_rate  
* n_mels - Number of mel filterbanks  
* f_min - Minimum frequency  
* f_max - Maximum frequency  
...


## My specs

RAM: 16GB  
GPU: 1660super 6GB  
CPU: i5-11400F  

## References

1. ERANNs: Efficient Residual Audio Neural Networks for Audio Pattern Recognition. `Sergey Verbitskiy`, `Vladimir Berikov`, `Viacheslav Vyshegorodtsev`. [arxiv abs](https://arxiv.org/abs/2106.01621v7)
2. PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition. `Qiuqiang Kong`, `Yin Cao`, `Turab Iqbal`, `Yuxuan Wang`, `Wenwu Wang`, `Mark D. Plumbley`. [arxiv abs](https://arxiv.org/abs/1912.10211)
