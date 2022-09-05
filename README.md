# Automated-Essay-Scoring-via-Pairwise-Contrastive-Regression

Created by Jiayi Xie*, Kaiwei Cai*, Li Kong, Junsheng Zhou, Weiguang Qu <br>
This repository contains the ASAP dataset and Pytorch implementation for Automated Essay Scoring.(Coling 2022, Oral) <br>

## Dataset

### ASAP
We use 5-corss-validation, and convert the dataset asap into 5 folds, as shown in the file path "./dataset/asap"

## Code for AES-NPCR

### Requirement

> - Pytorch 1.7.1
> - Python 3.7.9

### Pretrain Model

BERT, Roberta, XLNet can be used, default BERT

### Training

```commandline
# train a model on NPCR
# the number 1 and 0 means the Prompt 1 and the fold 0, and so on
nohup ./main.sh 768 1 0 bert &> ./{logs_path}/prompt1_fold0.logs &
```

****
The code will be refactored.
