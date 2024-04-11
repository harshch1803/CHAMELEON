# CHAMELEON: INCREASING LABEL-ONLY MEMBERSHIP LEAKAGE WITH ADAPTIVE POISONING
**Authors: Harsh Chaudhari, Giorgio Severi, Alina Oprea, Jonathan Ullman.**

Code for our [Chameleon: Increasing Label-Only Membership Leakage with Adaptive Poisoning](https://iclr.cc/virtual/2024/poster/19475) paper that will appear at ICLR 2024.

## Local Installation
This repository was developed on python version `3.9.13`. Please refer to `requirements.txt` for specific packages. 

## Chameleon Attack Code 

![Building Blocks](ChameleonBB.png)

Our attack works in three stages:

**I) Adaptive Poisoning Phase:** The script below trains resent-18 models for 500 challenge points for the dataset using our adaptive poisoning strategy and stores the progress in adapt.out file.

```shell
python3 -u train_out_models.py -d [--data] -out [--outmodels] -tp [--psnthresh] -kmax [--maxiters]  -c [--cuda]
```
The arguments for the script are as follows:
```shell
data (str) -- Dataset to run the attack. Options: 'cifar10', 'cifar100', 'gtsrb'.
outmodels (int) -- Number of OUT models. Default: '8'.
psnthresh (float) -- Poisoning Threshold. Default: '0.13'.
maxiters (int) -- Maximum Poisoning Iterations. Default: '8'.
cuda (int) -- The GPU device number to run the attack. Default: '0'.
```
The command below runs our attack with the default parameters used in our paper.
```shell
python3 -u train_out_models.py
```

**II) Membership Neighborhood Phase:** The script below find candidates for the 500 challenge point in the membership neighborhood space.
```shell
python3 -u find_neighbors.py -d [--data] -aug [--augrep] -tnb [--nbrthresh]  -c [--cuda]
```
The arguments for the script are as follows:
```shell
data (str) -- Dataset to run the attack. Options: 'cifar10', 'cifar100', 'gtsrb'.
augrep (int) -- Number of random augmentations to generate. Default: '128'.
nbrthresh (float) -- Poisoning Threshold. Default: '0.75'.
cuda (int) -- The GPU device number to run the attack. Default: '0'.
```

**III) Distinguishing Test:** The script below trains N separate target models which includes the poisoned set created from our Adaptive poisoning phase. 

```shell
python3 -u train_target_models.py -d [--data] -tgt [--tgtmodels]  -c [--cuda]
```

The arguments for the script are as follows:
```shell
data (str) -- Dataset to run the attack. Options: 'cifar10', 'cifar100', 'gtsrb'.
tgtmodels (int) -- Number of Target models. Default: '16'.
cuda (int) -- The GPU device number to run the attack. Default: '0'.
```

The script below runs our distinguishing test on the trained target models from above and saves the scores in 'SavedScores' folder. ROC curves are stored in the Figures folder.

```shell
python3 -u run_dtest.py -d [--data] -aug [--augrep] -c [--cuda]
```

The arguments for the script are as follows:
```shell
data (str) -- Dataset to run the attack. Options: 'cifar10', 'cifar100', 'gtsrb'.
augrep (int) -- Number of augmentations per challenge point. Default: '128'.
cuda (int) -- The GPU device number to run the attack. Default: '0'.