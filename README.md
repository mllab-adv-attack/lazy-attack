# Lazy-attack



## Ours implementations

### lls-imagenet

ImageNet version code of 'ours' method, paper version.
(admm versions to be implemented here)

Use this for ours on ImageNet.

### lls-cifar10

Cifar-10 version code of 'ours' method, paper version.
(admm versions to be implemented here)

Use this for ours on Cifar-10.

### l2-imagenet

L2 version of ours. Not used right now.



## Baseline implementations

### blackbox-attacks-bandits-priors

Code for baseline black-box attack 'Bandit' (ICLR 2019, https://arxiv.org/abs/1807.07978).
Adopted from https://github.com/MadryLab/blackbox-bandits.

Use this for Bandit results in ImageNet.

### bandit-imagenet

Code of blackbox-attacks-bandits-priors changed to use Tensorflow and match format of 'ours' code.

### nes-imagenet

Code for baseline block-box attack 'NES' (ICML 2018, https://arxiv.org/abs/1804.08598).
Adopted from https://github.com/labsix/limited-blackbox-attacks.

Use this for NES results in ImageNet.

## Others

### gaon_imagenet

Contains works before merging to a git repo.
Includes ImageNet PGD attack implementation adopted from https://github.com/MadryLab/cifar10_challenge.

Use this for PGD on ImageNet.

### gaon_cifar

Contains works before merging to a git repo.
Includes Cifar10 PGD attack implementation from https://github.com/MadryLab/cifar10_challenge.

Use this for PGD on Cifar-10.


## Downloading data

Cifar-10: cp from ~/../gaon/lazy-attack/cifar10_data to this folder.

ImageNet: cp from ~/../gaon/lazy-attack/imagenet_data to this folder.
