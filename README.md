# CFKGC_README

# Continual Few-shot Knowledge Graph Completion

CIKM 2024: "[Learning from Novel Knowledge: Continual Few-shot Knowledge Graph Completion](https://dl.acm.org/doi/pdf/10.1145/3627673.3679734)"

This repository contains the implementation of the paper:

> Zhuofeng Li, Haoxiang Zhang, Qiannan Zhang, Ziyi Kou, and Shichao Pei. 2024. Learning from Novel Knowledge: Continual Few-shot Knowledge Graph Completion. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management (CIKM '24). Association for Computing Machinery, New York, NY, USA, 1326–1335. https://doi.org/10.1145/3627673.3679734
> 

## Introduction

Knowledge graph (KG) completion is vital for uncovering missing knowledge and addressing incompleteness in KGs. While few-shot models following meta-learning have shown promise for rare relations, they typically assume static KGs. In reality, KGs evolve with newly emerging relations, requiring continual learning capabilities.

This work proposes a novel framework for continual few-shot KG completion that addresses two key challenges:

1. Catastrophic forgetting
2. Scarcity of novel relations

## Key Features

- Data-level triple rehearsal and model-level meta-learner modulation to combat forgetting
- Multi-view relation augmentation using self-supervised learning
- Enhanced generalization ability for limited novel relations

## Method Overview

![https://github.com/cfkgc-paper/CFKGC-paper/blob/main/imgs/overview.png](https://github.com/cfkgc-paper/CFKGC-paper/blob/main/imgs/overview.png)

## Getting Started

### Installation

```bash
git clone https://github.com/cfkgc-paper/CFKGC-paper.git
cd CFKGC-paper
pip install -r requirements.txt

```

### Data Preparation

The experiments use two datasets:

1. NELL-One [[Baidu Netdisk](https://pan.baidu.com/s/14ytl4goZCsVeWDIvmeTzeQ?pwd=gnn8)][[Google Drive(Pending...)]()]
2. Wiki-One [[Baidu Netdisk](https://pan.baidu.com/s/17-0rwDYHJPaW9sfKv_sfFg?pwd=74xz)][[Google Drive(Pending...)]()]

Total Zip with all codes: [[Baidu Netdisk]](https://pan.baidu.com/s/1Lo1a3KLMidLeNCTqZMCawA?pwd=pgnu)

The Structure of the project is as followings:  
>CFKGC-paper<br>
>&nbsp;&nbsp;&nbsp;&nbsp;|--./NELL  
>&nbsp;&nbsp;&nbsp;&nbsp;|--./Wiki  
>&nbsp;&nbsp;&nbsp;&nbsp;|--trainer.py  
>&nbsp;&nbsp;&nbsp;&nbsp;|--params.py  
>&nbsp;&nbsp;&nbsp;&nbsp;|--main.py  
>&nbsp;&nbsp;&nbsp;&nbsp;|--embedding.py  
>&nbsp;&nbsp;&nbsp;&nbsp;|--data_loader.py<br>
>&nbsp;&nbsp;&nbsp;&nbsp;|--...

Statistics of the datasets:

| Dataset | #Relation | #Entity | #Triples | #Task |
| --- | --- | --- | --- | --- |
| NELL-One | 358 | 68,545 | 181,109 | 67 |
| Wiki-One | 822 | 4,838,244 | 5,859,240 | 183 |

### Running the Code

The total train and evaluation process with reported hyper-parameters in the paper: 

```bash
# NELL-One
python main.py -data NELL-One -path ./NELL -if True -br 30 -bs 3 -nt 8 -l 0.1 -es_np 50
# Wiki-One
python main.py -data Wiki-One -path ./Wiki -if False -br 72 -bs 7 -nt 8 -l 1. -es_np 300
```

### Configuration

Extended setting can be set by altering the few-shot size, number-size here are key parameters in `params.py`:

```python
# Dataset setting
args.add_argument("-data", "--dataset", default="NELL-One", type=str)
args.add_argument("-path", "--data_path", default="./NELL", type=str)

# dataloader setting
# Support Numeber in Base Meta-learn stage
args.add_argument("-bfew", "--base_classes_few", default=3, type=int)
# Query Numeber in Base Meta-learn stage
args.add_argument("-bnq", "--base_classes_num_query", default=3, type=int)
# Support Numeber in Novel Tasks stage
args.add_argument("-few", "--few", default=3, type=int)
# Query Numeber in Novel Tasks stage
args.add_argument("-nq", "--num_query", default=3, type=int)
# Relation Numeber in Base Meta-learn stage
args.add_argument("-br", "--base_classes_relation", default=30, type=int)
# Novel Relations Numeber in each Novel Tasks
args.add_argument("-bs", "--batch_size", default=3, type=int)
# Number of Novel Tasks
args.add_argument("-nt", "--num_tasks", default=8, type=int)

```

## Citation

```
@inproceedings{10.1145/3627673.3679734,
author = {Li, Zhuofeng and Zhang, Haoxiang and Zhang, Qiannan and Kou, Ziyi and Pei, Shichao},
title = {Learning from Novel Knowledge: Continual Few-shot Knowledge Graph Completion},
year = {2024},
isbn = {9798400704369},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3627673.3679734},
doi = {10.1145/3627673.3679734},
pages = {1326–1335},
numpages = {10},
keywords = {continual learning, few-shot learning, knowledge graphs},
location = {Boise, ID, USA},
series = {CIKM '24}
}

```
