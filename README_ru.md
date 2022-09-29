# Deep Learning Survey for Audio Data

[русский](README_ru.md)

## Резюме

Обучение с учителем на датасете ESC50, с гибкой конфигурацией, основанное на мел спектрограммах и 2d свёртках. [Результаты](logs/results/esc50/)

## Датасеты

#### ESC-50

[![kaggle](https://www.kaggle.com/static/images/favicon.ico)](https://www.kaggle.com/datasets/ludovick/esc50dataset) [![paperswithcode](https://paperswithcode.com/favicon.ico)](https://paperswithcode.com/dataset/esc-50) [![github](https://github.githubassets.com/favicons/favicon-dark.svg)](https://github.com/karolpiczak/ESC-50)

ESC-50 датасет с **2000** природными аудио записями подходящий для сравнения методов аудио классификации.

Датасет состоит из **5**-секундных записей собранные в **50** классов (**40**  адуио на класс) состоящий из 5 основных категорий: `Animals`, `Natural soundscapes & water sounds`, `Human non-speech sounds`, `Interior/domestic sounds`, `Exterior/urban noises`

#### UrbanSound8K

[![kaggle](https://www.kaggle.com/static/images/favicon.ico)](https://www.kaggle.com/datasets/chrisfilo/urbansound8k) [![paperswithcode](https://paperswithcode.com/favicon.ico)](https://paperswithcode.com/dataset/urbansound8k-1)

Размечанный датасет с **8732** природными аудио запиясми (до **4** секуд) с **10** классами: `air_conditioner`, `car_horn`, `children_playing`, `dog_bark`, `drilling`, `enginge_idling`, `gun_shot`, `jackhammer`, `siren` and `street_music`

## Установка 🔨

Скачать ESC50 датасет, опционально UrbanSound8k

Настроить конфиги `config.yaml`

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

Доступные функции активаций: `gelu`, `relu`, `lrelu` (LeakyReLU), `swiglu` [4]   

sample_rate (int) - частота дискретизации проекта  

mel_spectrogram - словарь с аргументами для *torchaudio.transforms.MelSpectrogram* класса. Стандартные значения: *[1] 3.1. Feature Extraction*. про мел спектрограммы можно почитать здесь [habr](https://habr.com/ru/post/462527/) или здесь
* win_length - Размер окна в секундах, высчитывается по частоте дискретизации
* hop_length - Размер шага между окнами в секундах, высчитывается по частоте дискретизации
* n_mels - Количество мел-фильтров
* f_min - Минимальная частота
* f_max - Максимальная частота

augmentation - аугментация данных. Аугментация позволяет увеличить разнообразие наших данных, применяя к исходным данным разного рода преобразования. Каждая аугментация имеет параметр `p` - вероятность включения этой трансформации для каждого отдельно взятого семпла/батча. Укажити p=0 для отключения определённой трансформации. Часть кода взята из: *[3] 3.2. Data Augmentation*.
* reverb: реверберация, не стоит её использовать, крайне медленная операция свёртки
  * p (float) - вероятность включения трансформации на каждый батч
* time_shift: случайно разделяет аудио на две части и сулчайно растягивает/сжимает каждую
  * p (float) - вероятность
* noise: генерирует и добавляет шум к сигналу
  * p (float) - вероятность
  * min_snr_db (float) - минимальная громкость генерируемого шума по отношению к исходному сигналу
  * max_snr_db (float) - максимальная громкость генерируемого шума по отношению к исходному сигналу
* dropblock: зануляет прямоугольные части картинки (спектрограммы) *[6] - DropBlock: A regularization method for convolutional networks*
  * p  (float) - вероятность включения трансформации на каждый батч
  * dp (float) - количество блоков в %, аналогично `p` в классическом `Dropout`
  * block_size:
    * t (int) - размер блока вдоль временной оси
    * f (int) - размер блока вдоль частотной оси
* time_masking: маскирует фрагмент спектрограммы по оси времени
  * p (float) - вероятность
  * tmp (int) - `time_mask_param`, максимальная длина маски.
  * num (int) - количество масок
* freq_masking: маскирует фрагмент спектрограммы по частотной оси
  * p (float) - вероятность
  * fmp (int) - `freq_mask_param`, максимальная длина маски.
  * num (int) - количество масок

pretrain_urban (bool) - предобучение модели на датасете urbansound  

pretrain - параметры предобучения
* num_workers (int) - количество подпроцессов используемых для загрузки данных. 0 означает что данные будут загружаться в основном процессе. Не рекуомендуется использовать количество больше физических ядер на процессоре 
* epochs (int) - количество эпох. Если 0 и `patience` > 0 будет обучаться пока 
* lr (float) - коэффициент скорости обучения
* weight_decay (float) - weight decay коэффициент [10]
* batch_size (int) - размер батча
* accum_iter (int) - количество gradient accumulation. Должно быть больше  [7]
* grad_clip_value (float) - значение gradient clipping norm [8]
* grad_clip_norm (float) - значение gradient clipping [8]
* use_checkpoint (bool) - gradient checkpoint. не стоит это использовать [9]

train - параметры обучения на ESC50, аналогично pretrain  

model - особенности и архитектура модели. Входная размерность: (B, C, T, F) где B - размер батча, T - временной размер (зависит от длины аудио), F - частотный размер, зависит от параметров алгоритма мел спектрограммы 
* res_block - конфигурация для всех конволюционных-остаточных блоков, с 3 конволюционными слоями и 2 батч нормализациями.
  * activation (str) - функция активации между конволюционными слоями
  * batch_first (bool) - применяет батч нормализацию после функции активации если `true` и перед активацией если `false`
  * kernel_size (list[int, int]) - размер ядра двух основынх свёрток
  * kernel_size_res (list[int, int]) - размер ядра остаточной свёртки
* blocks (dictionary[str, list|int|float]) - архитектура модели. Список с блоками и слоями. Доступные блоки и слои:
  * res (list[int, int]) - конволюционно-остаточный блок, список с двумя значениями: количество фильтров первого конволюционного слоя и второго
  * max_pool (list[int, int]) - слой пулинга, список с двумя значениями: размер ядра по оси T (всегда равно 1 по оси F) и шаг по оси T (всегда равно 1 по оси F)
  * activation (str) - функция активации
  * drop (list[float, float, int, int]) - dropblock слой, список с 4 значениями: вероятность включения трансформации на каждый батч, количество блоков в %, размер блока по оси T, размер блока по оси F
  * dropout (float) - коэфициент dropout2d
  * fc (int) - Линейный слой, целочисленное значение количество выходных нейронов
  * pooling - слой пулинга
    * mode (str) - доступные режимы пулинга:
      * avg - average pooling
      * max - max pooling
      * sum - sum of avg and max pooling
      * mul - mul of avg and max pooling
    * flat (bool) - развернуть вектор по оси C и F. На входе (B, C, T, F) на выходе (B, T, F*C)

## Источники

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