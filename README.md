# Position-aware Graph Neural Networks
This repository is the official PyTorch implementation of "Position-aware Graph Neural Networks".

[Jiaxuan You](https://cs.stanford.edu/~jiaxuan/), [Rex Ying](https://cs.stanford.edu/people/rexy/), [Jure Leskovec](https://cs.stanford.edu/people/jure/index.html), [Position-aware Graph Neural Networks](http://proceedings.mlr.press/v97/you19b/you19b.pdf), ICML 2019 (long oral).

## Installation

- Install PyTorch (tested on 1.0.0), please refer to the offical website for further details
```bash
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
```
- Install PyTorch Geometric (tested on 1.1.2), please refer to the offical website for further details
```bash
pip install --verbose --no-cache-dir torch-scatter
pip install --verbose --no-cache-dir torch-sparse
pip install --verbose --no-cache-dir torch-cluster
pip install --verbose --no-cache-dir torch-spline-conv (optional)
pip install torch-geometric
```
- Install networkx (tested on 2.3), make sure you are not using networkx 1.x version!
```bash
pip install networkx
```
- Install tensorboardx
```bash
pip install tensorboardX
```
- If you wish to use PPI dataset, unzip `data/ppi.zip`


## Run
- 3-layer GCN, grid
```bash
python main.py --model GCN --layer_num 3 --dataset grid
```
- 2-layer P-GNN, grid
```bash
python main.py --model PGNN --layer_num 2 --dataset grid
```
- 2-layer P-GNN, grid, with 2-hop shortest path distance
```bash
python main.py --model GCN --layer_num 2 --approximate 2 --dataset grid
```
- 3-layer GCN, all datasets
```bash
python main.py --model GCN --layer_num 3 --dataset All
```
- 2-layer PGNN, all datasets
```bash
python main.py --model PGNN --layer_num 2 --dataset All
```
You are highly encouraged to tune all kinds of hyper-parameters to get better performance. We only did very limited hyper-parameter tuning.

We recommend using tensorboard to monitor the training process. To do this, you may run
```bash
tensorboard --logdir runs
```