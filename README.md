Keras CNN Sample
====
## Description
Kerasを使ったCNNにより画像認識プログラムのトレーニングと推論を行うプログラム。
できるだけシンプルになるように作りました。

## Install (Using Anaconda)
### Install Anaconda
```
$ wget https://repo.continuum.io/archive/Anaconda2-4.3.1-Linux-x86_64.sh

$ bash Anaconda2-4.3.1-Linux-x86_64.sh
$ source ~/.bashrc
```

### Create Env
```
$ conda create --name nn_sample python=2.7
$ source activate nn_sample
$ pip install Keras==1.2.0
$ pip install numpy==1.11.2
$ pip install tensorflow==1.1.0
$ pip install opencv-python==3.2.0.7
$ pip install h5py==2.7.0
```

### Add Train & Test Images
```
$ mkdir -p train_images/cat/ train_images/dog/
$ mkdir -p test_images/cat/ test_images/dog/
```
* train images into 「train_images/cat or dog」
* test images into 「test_images/cat or dog」

## Usage
### Traning
```
# 学習した結果を保存
$ python sample.py -o ./weights/my_model_weights.h5

# 学習結果を読み込み、再学習させてから結果を保存
$ python sample.py -w ./weights/my_model_weights.h5 -o ./weights/my_model_weights.h5
```

### Predict
```
# 推論のみ実行
$ python sample.py -P -w ./weights/my_model_weights.h5
```

### Add Category
1. Add NewFolder to「test_images/」and 「train_images」
2. Add Images Into NewFolder
3. Add Labels to 「sample.py > create_labels(images_path):」

## Licence

[MIT](https://github.com/hirasaki1985/Keras_cnn_sample/blob/master/LICENSE)

## Author

[mamoru hirasaki](https://github.com/hirasaki1985)
