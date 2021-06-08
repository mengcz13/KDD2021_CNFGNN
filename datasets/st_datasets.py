import os
import pickle
from functools import partial

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_pems_data(rootpath, name, adj_mx_name, ratio=1.0, one_node=False, one_sample=False):
    if ratio < 1.0:
        inductive = True
        partial_nodes = np.load(os.path.join(rootpath, 'sensor_graph', '{}_partial_nodes.npz'.format(name)), allow_pickle=True)
        partial_nodes = partial_nodes[str(ratio)]
        selected_nodes, _ = partial_nodes
    else:
        inductive = False

    adj_mx_path = os.path.join(rootpath, 'sensor_graph', adj_mx_name)
    _, _, adj_mx = load_pickle(adj_mx_path)
    adj_mx_ts = torch.from_numpy(adj_mx).float()
    if inductive:
        train_adj_mx_ts = adj_mx_ts[selected_nodes, :][:, selected_nodes]
        eval_adj_mx_ts = adj_mx_ts
    else:
        train_adj_mx_ts, eval_adj_mx_ts = adj_mx_ts, adj_mx_ts
    train_edge_index, train_edge_attr = dense_to_sparse(train_adj_mx_ts)
    eval_edge_index, eval_edge_attr = dense_to_sparse(eval_adj_mx_ts)

    datapath = os.path.join(rootpath, name)
    raw_data = {}
    for name in ['train', 'val', 'test']:
        raw_data[name] = np.load(os.path.join(datapath, '{}.npz'.format(name)))

    if inductive:
        selected_ts = torch.BoolTensor([False] * eval_adj_mx_ts.shape[0])
        selected_ts[selected_nodes] = True
        selected_ts = selected_ts.unsqueeze(-1)

    FEATURE_START, FEATURE_END = 0, 1
    ATTR_START, ATTR_END = 1, 2

    train_features = raw_data['train']['x'][..., FEATURE_START:FEATURE_END]
    if inductive:
        train_features = train_features[:, :, selected_nodes, :]
    train_features = train_features.reshape(-1, train_features.shape[-1])
    feature_scaler = StandardScaler(
        mean=train_features.mean(axis=0), std=train_features.std(axis=0)
    )
    attr_scaler = StandardScaler(
        mean=0, std=1
    )
    loaded_data = {
        'feature_scaler': feature_scaler,
        'attr_scaler': attr_scaler
    }

    for name in ['train', 'val', 'test']:
        x = feature_scaler.transform(raw_data[name]['x'][..., FEATURE_START:FEATURE_END])
        y = feature_scaler.transform(raw_data[name]['y'][..., FEATURE_START:FEATURE_END])
        x_attr = attr_scaler.transform(raw_data[name]['x'][..., ATTR_START:ATTR_END])
        y_attr = attr_scaler.transform(raw_data[name]['y'][..., ATTR_START:ATTR_END])

        # for debugging
        if one_node:
            x = x[:, :, 0:1, :]
            y = y[:, :, 0:1, :]
            x_attr = x_attr[:, :, 0:1, :]
            y_attr = y_attr[:, :, 0:1, :]
        if one_sample:
            x, y, x_attr, y_attr = x[0:1], y[0:1], x_attr[0:1], y_attr[0:1]

        data = {}
        if name is 'train':
            edge_index, edge_attr = train_edge_index, train_edge_attr
        else:
            edge_index, edge_attr = eval_edge_index, eval_edge_attr
        data.update(
            x=torch.from_numpy(x).float(), y=torch.from_numpy(y).float(),
            x_attr=torch.from_numpy(x_attr).float(),
            y_attr=torch.from_numpy(y_attr).float(),
            edge_index=edge_index, edge_attr=edge_attr
        )
        if name is 'train' and inductive:
            data.update(selected=selected_ts)
        loaded_data[name] = data

    return loaded_data


def load_hetero_nri_sim_data(rootpath, name):
    datapath = rootpath
    suffix = '_' + name
    loc_train = np.load(os.path.join(datapath, 'loc_train' + suffix + '.npy'))
    vel_train = np.load(os.path.join(datapath, 'vel_train' + suffix + '.npy'))
    edges_train = np.load(os.path.join(datapath, 'edges_train' + suffix + '.npy'))

    loc_valid = np.load(os.path.join(datapath, 'loc_valid' + suffix + '.npy'))
    vel_valid = np.load(os.path.join(datapath, 'vel_valid' + suffix + '.npy'))
    edges_valid = np.load(os.path.join(datapath, 'edges_valid' + suffix + '.npy'))

    loc_test = np.load(os.path.join(datapath, 'loc_test' + suffix + '.npy'))
    vel_test = np.load(os.path.join(datapath, 'vel_test' + suffix + '.npy'))
    edges_test = np.load(os.path.join(datapath, 'edges_test' + suffix + '.npy'))

    # [num_samples, num_timesteps, num_dims, num_atoms]
    num_atoms = loc_train.shape[3]

    loc_max = loc_train.max()
    loc_min = loc_train.min()
    vel_max = vel_train.max()
    vel_min = vel_train.min()

    # Normalize to [-1, 1]
    loc_train = (loc_train - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_train = (vel_train - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_valid = (loc_valid - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_valid = (vel_valid - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_test = (loc_test - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_test = (vel_test - vel_min) * 2 / (vel_max - vel_min) - 1

    # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    loc_train = np.transpose(loc_train, [0, 3, 1, 2])
    vel_train = np.transpose(vel_train, [0, 3, 1, 2])
    feat_train = np.concatenate([loc_train, vel_train], axis=3)
    edges_train = np.reshape(edges_train, [-1, num_atoms ** 2])
    edges_train = np.array((edges_train + 1) / 2, dtype=np.int64)
    edges_train = np.reshape(edges_train, [-1, num_atoms, num_atoms])

    loc_valid = np.transpose(loc_valid, [0, 3, 1, 2])
    vel_valid = np.transpose(vel_valid, [0, 3, 1, 2])
    feat_valid = np.concatenate([loc_valid, vel_valid], axis=3)
    edges_valid = np.reshape(edges_valid, [-1, num_atoms ** 2])
    edges_valid = np.array((edges_valid + 1) / 2, dtype=np.int64)
    edges_valid = np.reshape(edges_valid, [-1, num_atoms, num_atoms])

    loc_test = np.transpose(loc_test, [0, 3, 1, 2])
    vel_test = np.transpose(vel_test, [0, 3, 1, 2])
    feat_test = np.concatenate([loc_test, vel_test], axis=3)
    edges_test = np.reshape(edges_test, [-1, num_atoms ** 2])
    edges_test = np.array((edges_test + 1) / 2, dtype=np.int64)
    edges_test = np.reshape(edges_test, [-1, num_atoms, num_atoms])

    feat_train = torch.FloatTensor(feat_train)
    edges_train = torch.LongTensor(edges_train) # B x N x N
    feat_valid = torch.FloatTensor(feat_valid)
    edges_valid = torch.LongTensor(edges_valid)
    feat_test = torch.FloatTensor(feat_test)
    edges_test = torch.LongTensor(edges_test)

    # change to B x T x N x F
    feat_train = feat_train.permute(0, 2, 1, 3)
    feat_valid = feat_valid.permute(0, 2, 1, 3)
    feat_test = feat_test.permute(0, 2, 1, 3)

    # Exclude self edges
    # off_diag_idx = np.ravel_multi_index(
    #     np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
    #     [num_atoms, num_atoms])
    # edges_train = edges_train[:, off_diag_idx]
    # edges_valid = edges_valid[:, off_diag_idx]
    # edges_test = edges_test[:, off_diag_idx]

    feature_scaler = StandardScaler(mean=0, std=1)
    attr_scaler = StandardScaler(
        mean=0, std=1
    )
    loaded_data = {
        'feature_scaler': feature_scaler,
        'attr_scaler': attr_scaler
    }

    def batch_dense_to_sparse(batch_dense):
        batch_edge_index = []
        batch_edge_attr = []
        for t in range(batch_dense.shape[0]):
            edge_index, edge_attr = dense_to_sparse(batch_dense[t])
            batch_edge_index.append(edge_index)
            batch_edge_attr.append(edge_attr)
        return batch_edge_index, batch_edge_attr

    edge_index_train, edge_attr_train = batch_dense_to_sparse(edges_train)
    edge_index_val, edge_attr_val = batch_dense_to_sparse(edges_valid)
    edge_index_test, edge_attr_test = batch_dense_to_sparse(edges_test)

    loaded_data['train'] = {
        'x': feat_train[:, 0:20, :, :],
        'y': feat_train[:, 20:40, :, :],
        'x_attr': feat_train[:, 0:20, :, :0],
        'y_attr': feat_train[:, 20:40, :, :0],
        'edge_index': edge_index_train,
        'edge_attr': edge_attr_train
    }
    loaded_data['val'] = {
        'x': feat_valid[:, 0:20, :, :],
        'y': feat_valid[:, 20:40, :, :],
        'x_attr': feat_valid[:, 0:20, :, :0],
        'y_attr': feat_valid[:, 20:40, :, :0],
        'edge_index': edge_index_val,
        'edge_attr': edge_attr_val
    }
    loaded_data['test'] = {
        'x': feat_test[:, 0:20, :, :],
        'y': feat_test[:, 20:40, :, :],
        'x_attr': feat_test[:, 0:20, :, :0],
        'y_attr': feat_test[:, 20:40, :, :0],
        'edge_index': edge_index_test,
        'edge_attr': edge_attr_test
    }

    return loaded_data


available_datasets = {
    'METR-LA': partial(load_pems_data, rootpath='data/traffic/data',
        name='METR-LA', adj_mx_name='adj_mx.pkl'),
    'METR-LA-onenode': partial(load_pems_data, rootpath='data/traffic/data',
        name='METR-LA', adj_mx_name='adj_mx.pkl', one_node=True, one_sample=False),
    'METR-LA-onesample': partial(load_pems_data, rootpath='data/traffic/data',
        name='METR-LA', adj_mx_name='adj_mx.pkl', one_node=False, one_sample=True),
    'METR-LA-0.25': partial(load_pems_data, rootpath='data/traffic/data',
        name='METR-LA', adj_mx_name='adj_mx.pkl', ratio=0.25),
    'METR-LA-0.5': partial(load_pems_data, rootpath='data/traffic/data',
        name='METR-LA', adj_mx_name='adj_mx.pkl', ratio=0.5),
    'METR-LA-0.75': partial(load_pems_data, rootpath='data/traffic/data',
        name='METR-LA', adj_mx_name='adj_mx.pkl', ratio=0.75),
    'METR-LA-0.05': partial(load_pems_data, rootpath='data/traffic/data',
        name='METR-LA', adj_mx_name='adj_mx.pkl', ratio=0.05),
    'METR-LA-0.9': partial(load_pems_data, rootpath='data/traffic/data',
        name='METR-LA', adj_mx_name='adj_mx.pkl', ratio=0.9),
    'PEMS-BAY': partial(load_pems_data, rootpath='data/traffic/data',
        name='PEMS-BAY', adj_mx_name='adj_mx_bay.pkl'),
    'PEMS-BAY-0.25': partial(load_pems_data, rootpath='data/traffic/data',
        name='PEMS-BAY', adj_mx_name='adj_mx_bay.pkl', ratio=0.25),
    'PEMS-BAY-0.5': partial(load_pems_data, rootpath='data/traffic/data',
        name='PEMS-BAY', adj_mx_name='adj_mx_bay.pkl', ratio=0.5),
    'PEMS-BAY-0.75': partial(load_pems_data, rootpath='data/traffic/data',
        name='PEMS-BAY', adj_mx_name='adj_mx_bay.pkl', ratio=0.75),
    'PEMS-BAY-0.05': partial(load_pems_data, rootpath='data/traffic/data',
        name='PEMS-BAY', adj_mx_name='adj_mx_bay.pkl', ratio=0.05),
    'PEMS-BAY-0.9': partial(load_pems_data, rootpath='data/traffic/data',
        name='PEMS-BAY', adj_mx_name='adj_mx_bay.pkl', ratio=0.9),
    'NRI_springs5': partial(load_hetero_nri_sim_data, rootpath='data/NRI',
        name='springs5')
}


def load_dataset(dataset_name):
    return available_datasets[dataset_name]()
