# Parsimonious Black-Box Adversarial Attacks via Efficient Combinatorial Optimization

## Environment
- Ubuntu 16.04
- python 3.5
- tensorflow 1.4.0 gpu
- CUDA 8.0
- cuDNN 6.0

## Prerequisites

### Cifar-10
1. Download an adversarially pretrained model from Madry's CIFAR10 Adversarial Examples Challenge, and set `MODEL_DIR` in main.py to the location of the downloaded model.
2. Download Cifar-10 dataset, decompress it, and set `DATA_DIR` in main.py to the location of the dataset.

### ImageNet
1. Download a pretrained Inception v3 model from https://github.com/tensorflow/models/tree/master/research/slim, decompress it, and place it to `tools/data` folder.
2. Download ImageNet validation set (should contain a folder named `val` and a file named `val.txt`), and set `IMAGENET_PATH` in main.py to the location of the dataset.

## How to Run

### Cifar-10
1. For untargeted attack, 
`python main.py --epsilon 8 --max_queries 20000`

### ImageNet
1. For untargeted attack,
`python main.py --epsilon 0.05 --max_queries 10000`
2. For targeted attack,
 `python main.py --targeted --epsilon 0.05 --max_queries 100000`

## Arguments
- `img_index_start`: Given a list of image numbers for evaluation, it defines the starting index of the list.
- `sample_size`: The number of images to be attacked.
- `epsilon`: The maximum distortion of a pixel value.
- `loss_func`: A type of loss function, xent (Cross entropy loss) or cw (Carlini-Wagner loss). 

## Baseline methods
- We reproduce NES and Bandits for experiments on Cifar-10.
- To run NES and Bandits, download model and dataset (same as above), and go to each directory and run `python main.py`.
