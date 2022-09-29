# Deep Learning Survey for Audio Data

[README на русском](README_ru.md)

## Abstract

Supervised train pipeline, with flexible configuratio, based on mel spectrograms and 2d convolution. [Results](logs/results/esc50/)

## Datasets

#### ESC-50

[![kaggle](https://www.kaggle.com/static/images/favicon.ico)](https://www.kaggle.com/datasets/ludovick/esc50dataset) [![paperswithcode](https://paperswithcode.com/favicon.ico)](https://paperswithcode.com/dataset/esc-50) [![github](https://github.githubassets.com/favicons/favicon-dark.svg)](https://github.com/karolpiczak/ESC-50)

The ESC-50 dataset is a labeled collection of **2000** environmental audio recordings suitable for benchmarking methods of environmental sound classification.

The dataset consists of **5**-second-long recordings organized into **50** semantical classes (with **40** examples per class) loosely arranged into 5 major categories: `Animals`, `Natural soundscapes & water sounds`, `Human non-speech sounds`, `Interior/domestic sounds`, `Exterior/urban noises`

#### UrbanSound8K

[![kaggle](https://www.kaggle.com/static/images/favicon.ico)](https://www.kaggle.com/datasets/chrisfilo/urbansound8k) [![paperswithcode](https://paperswithcode.com/favicon.ico)](https://paperswithcode.com/dataset/urbansound8k-1)

This dataset contains **8732** labeled environmental sound excerpts (<=**4s**) of urban sounds from **10** classes: `air_conditioner`, `car_horn`, `children_playing`, `dog_bark`, `drilling`, `enginge_idling`, `gun_shot`, `jackhammer`, `siren` and `street_music`

## Setup 🔨

Download ESC50 and optionally UrbanSound8k

Configure `config.yaml`

```
pip install requirements.txt
```

```
python preproccesing.py
```

```
python train.py
```

## My specs

RAM: 16GB
GPU: 1660super 6GB
CPU: i5-11400F

## Config

avaible activation: `gelu`, `relu`, `lrelu` (LeakyReLU), `swiglu` [4]  

sample_rate (int) - project sasmple rate, default 44_100  

mel_spectrogram - dict with argumetns for *torchaudio.transforms.MelSpectrogram* class. Default param: *[1] 3.1. Feature Extraction*.
* win_length - Window size in sec, calculated by sample_rate
* hop_length - Length of hop between STFT windows in sec, calculated by sample_rate
* n_mels - Number of mel filterbanks
* f_min - Minimum frequency
* f_max - Maximum frequency

augmentation - data augmentation config. Each augmentation technique has parametr `p` - probability to include this augmentation on every single sample. Set p to 0 to disable specific aug. Some of the code is from: *[3] 3.2. Data Augmentation*.
* reverb: reverberation, don't use this thing, it's very slow
  * p (float) - probability
* time_shift: randomly split audio and randomly stretches/compresses each part
  * p (float) - probability
* noise: adds noise
  * p (float) - probability
  * min_snr_db (float) - min noise db
  * max_snr_db (float) - max noise db
* dropblock: *[6] - DropBlock: A regularization method for convolutional networks*
  * p  (float) - probability
  * dp (float) - dropblock probability, similar to `p` in `Dropout`
  * block_size:
    * t (int) - block size along time axis
    * f (int) - block size along freq. axis
* time_masking: masking fragment of sample by time axis
  * p (float) - probability
  * tmp (int) - `time_mask_param`, maximum possible length of the mask. Indices uniformly sampled from [0, freq_mask_param)
  * num (int) - num of mask
* freq_masking: masking fragment of sample by freq. axis
  * p (float) - probability
  * fmp (int) - `freq_mask_param`, maximum possible length of the mask. Indices uniformly sampled from [0, freq_mask_param)
  * num (int) - num of mask

pretrain_urban (bool) - pretrain model on urban dataset  

pretrain - pretrain options  

* num_workers (int) - how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process
* epochs (int) - num of train epoch. if 0 and `patience` > 0 will train until the best score on validation set
* lr (float) - learning rate
* weight_decay (float) - weight decay coefficient [10]
* batch_size (int) - batch size
* accum_iter (int) - num of gradient accumulation. Must be greater than 0 [7]
* grad_clip_value (float) - gradient clipping norm value [8]
* grad_clip_norm (float) - gradient clipping value [8]
* use_checkpoint (bool) - gradient checkpoint. memory leak, dont use it [9]

train - train on esc50 dataset options, same as pretrain.  

model - model specifics and architecture. Notice input shape is (B, C, T, F) where B - batch size, T - temporal size, F - frequency size
* res_block - config for all residual convolution block, with 3 conv layer and 2 batchnorms.
  * activation (str) - activation between conv layers, see avaible activation
  * batch_first (bool) - apply batchnorm before activation if `true` after activation if `false`
  * kernel_size (list[int, int]) - kernel size of two main convolution
  * kernel_size_res (list[int, int]) - kernel size of residual convolution
* blocks (dictionary[str, list|int|float]) - model architecture. list with block. avaible blocks:
  * res (list[int, int]) - residual convolution block, list with two int values: num of first conv layer filters and num of second conv layer filters
  * max_pool (list[int, int]) - max pooling layer,  list with two int values: kernel size over T dim. (always 1 over F dim.) and stride over T dim. (always 1 over F dim.)
  * activation (str) - activation layer, see avaible activation
  * drop (list[float, float, int, int]) - dropblock layer, list with values: p propability, dropblock probability, block_size over T, block_size over F
  * dropout (float) - dropout2d, float dropout rate value
  * fc (int) - Linear layer, int value - out_features
  * pooling - global pooling layer
    * mode (str) - pooling mode:
      * avg - average pooling
      * max - max pooling
      * sum - sum of avg and max pooling
      * mul - mul of avg and max pooling
    * flat (bool) - flatten over C and F

## References

1. ERANNs: Efficient Residual Audio Neural Networks for Audio Pattern Recognition. *Sergey Verbitskiy*, *Vladimir Berikov*, *Viacheslav Vyshegorodtsev*. [arxiv abs](https://arxiv.org/abs/2106.01621v7)
2. PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition. *Qiuqiang Kong*, *Yin Cao*, *Turab Iqbal*, *Yuxuan Wang*, *Wenwu Wang*, *Mark D. Plumbley*. [arxiv abs](https://arxiv.org/abs/1912.10211)
3. End-to-End Audio Strikes Back: Boosting Augmentations Towards An Efficient Audio Classification Network. *Avi Gazneli*, *Gadi Zimerman*, *Tal Ridnik*, *Gilad Sharir*, *Asaf Noy*. [arxiv abs](https://arxiv.org/abs/2204.11479)
4. GLU Variants Improve Transformer. *Noam Shazeer*. [arxiv abs](https://arxiv.org/abs/2002.05202v1)
5. When Does Label Smoothing Help? *Rafael Müller*, *Simon Kornblith*, *Geoffrey Hinton*. [arxiv abs](https://arxiv.org/abs/1906.02629)
6. DropBlock: A regularization method for convolutional networks. *Golnaz Ghiasi*, *Tsung-Yi Lin*, *Quoc V. Le*. [arxiv abs](https://arxiv.org/abs/1810.12890v1)
7. What is Gradient Accumulation in Deep Learning? [towardsdatascience](https://towardsdatascience.com/what-is-gradient-accumulation-in-deep-learning-ec034122cfa)
8. Gradient Clipping. [paperswithcode](https://paperswithcode.com/method/gradient-clipping)
9. Gradient Checkpointing. [paperswithcode](https://paperswithcode.com/method/gradient-checkpointing)
10. Understanding and Scheduling Weight Decay. *Zeke Xie*, *Issei Sato*, *Masashi Sugiyama*.[arxiv abs](https://arxiv.org/abs/2011.11152)