# GraphFM
Official code for "[GraphFM: A Comprehensive Benchmark for Graph Foundation Model](https://arxiv.org/abs/2406.08310)". GraphFM is a comprehensive benchmark for Graph Foundation Models (Graph FMs) and is based on graph self-supervised learning (GSSL). It aims to research the homogenization and scalability of Graph FMs. 

## Overview of the Benchmark
GraphFM provides a fair and comprehensive platform to evaluate existing GSSL works and facilitate future research.

![architecture](https://github.com/NYUSHCS/GraphFM/blob/main/img/architecture.png)

We perform a comprehensive benchmark of state-of-the-art self-supervised GNN models through four key aspects: dataset scale, training strategies, GSSL methods for Graph FMs, and adaptability to different downstream tasks.

## Installation
The required packages can be installed by running `pip install -r requirements.txt`.

## 🚀Quick Start

### Set up Model, Dataset and Batch type parameters

**model** ： 
`BGRL`, `CCA-SSG`, `GBT`, `GCA`, `GraphECL`, `GraphMAE`, `GraphMAE2`, `S2GAE`

**dataset** ： 
`cora`, `pubmed`, `citeseer`, `Flickr`, `Reddit`, `ogbn-arxiv`

**batch_type** ：
`full_batch`, `node_sampling`, `subgraph_sampling`

### Get Best Hyperparameters
You can run the `python main_optuna.py --type_model $model --dataset $dataset --batch_type $batch_type` to get the best hyperparameters.

### Train the Main Code
You can train the model with `main.py` after obtaining the hyperparameters tuned by Optuna.

## 📱️Updates
2024.6.15 Submitted our paper to arXiv.

## Reference

| **ID** | **Paper** | **Method** | **Conference** |
|--------|---------|:----------:|:--------------:|
| 1      | [Large-Scale Representation Learning on Graphs via Bootstrapping](https://arxiv.org/abs/2102.06514)      |    BGRL     |   ICLR 2022    |
| 2      | [From Canonical Correlation Analysis to Self-supervised Graph Neural Networks](https://arxiv.org/abs/2106.12484) |    CCA-SSG     |   NeurIPS 2021    |
| 3      | [Graph Barlow Twins: A self-supervised representation learning framework for graphs](https://arxiv.org/abs/2106.02466)  |   GBT   |    Knowledge-Based Systems 2022    |
| 4      | [Graph Contrastive Learning with Adaptive Augmentation](https://arxiv.org/abs/2010.14945)  |    GCA    |  WWW 2021  |
| 5      | [GraphECL: Towards Efficient Contrastive Learning for Graphs](https://github.com/GraphECL/GraphECL)  |    GraphECL    | Under Review |
| 6      | [GraphMAE: Self-Supervised Masked Graph Autoencoders](https://arxiv.org/abs/2205.10803)  |  GraphMAE   |   KDD 2022    |
| 7      | [GraphMAE2: A Decoding-Enhanced Masked Self-Supervised Graph Learner](https://arxiv.org/abs/2304.04779)  |   GraphMAE2    |   WWW 2023    |
| 8      | [S2GAE: Self-Supervised Graph Autoencoders are Generalizable Learners with Graph Masking](https://dl.acm.org/doi/abs/10.1145/3539597.3570404) |   S2GAE    |    WSDM 2023    |
