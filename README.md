# Cross-Node Federated Graph Neural Network for Spatio-Temporal Data Modeling

This repository is the official PyTorch implementation of "Cross-Node Federated Graph Neural Network for Spatio-Temporal Modeling".

## Setup

### Environment

```bash
conda create -n fedgnn "python<3.8"
conda activate fedgnn
bash install.sh
```

### Data

Download [`data.tar.bz2`](https://zenodo.org/record/4521262/files/data.tar.bz2?download=1). Then extract it to the root directory of the repository:

```bash
tar -xjf data.tar.bz2
```

## Experiments

### Main Experiments

`submission_exps/exp_main.sh` contains all commands used for experiments in Table 2 and Table 3.

### Inductive Learning on Unseen Nodes

Run `python submission_exps/exp_inductive.py` to print all commands for Table 4.

### Ablation Study: Effect of Alternating Training of Node-Level and Spatial Models

Run `python submission_exps/exp_at.py` to print all commands for Figure 2.

### Ablation Study: Effect of Client Rounds and Server Rounds

Run `python submission_exps/exp_crsr.py` to print all commands for Figure 3.
