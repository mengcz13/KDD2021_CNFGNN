# federated multi-task learning: model error + graph Laplacian regularization (https://arxiv.org/pdf/1705.10467.pdf)
import os
from argparse import ArgumentParser
from multiprocessing import cpu_count
from copy import deepcopy
from collections import defaultdict
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torch_geometric.data import DataLoader, Data
from torch.utils.data import TensorDataset
from tqdm import tqdm

from datasets.st_datasets import load_dataset
import models.base_models as base_models
from models.st_prediction.standalone import unscaled_metrics


class FedMTLNodePredictorClient(nn.Module):
    def __init__(self, base_model_name, optimizer_name,
        train_dataset, val_dataset, test_dataset, feature_scaler,
        sync_every_n_epoch, lr, weight_decay, mtl_lambda, batch_size, client_device, start_global_step,
        *args, **kwargs):
        super().__init__()
        self.base_model_name = base_model_name
        self.optimizer_name = optimizer_name
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.feature_scaler = feature_scaler
        self.sync_every_n_epoch = sync_every_n_epoch
        self.lr = lr
        self.weight_decay = weight_decay
        self.mtl_lambda = mtl_lambda
        self.batch_size = batch_size
        self.base_model_kwargs = kwargs
        self.device = client_device

        self.base_model_class = getattr(base_models, self.base_model_name)
        self.init_base_model(None)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        if self.val_dataset:
            self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            self.val_dataloader = self.train_dataloader
        if self.test_dataset:
            self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            self.test_dataloader = self.train_dataloader

        self.global_step = start_global_step

    def forward(self, x):
        return self.base_model(x, self.global_step)

    def init_base_model(self, state_dict):
        self.base_model = self.base_model_class(**self.base_model_kwargs).to(self.device)
        if state_dict is not None:
            self.base_model.load_state_dict(state_dict)
        self.optimizer = getattr(torch.optim, self.optimizer_name)(self.base_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def _get_flattened_model_params(self, model_named_params):
        named_params = [(name, p) for name, p in model_named_params]
        named_params = sorted(named_params, key=lambda x: x[0])
        flattend_params = torch.cat([x[1].flatten() for x in named_params])
        return flattend_params

    def local_train(self, state_dict_to_load):
        # state_dict_to_load: list of (adj_mx_weight_ij, state_dict_client_j). 0 is always for self state_dict.
        weight_list, state_dict_list = list(zip(*state_dict_to_load))
        if state_dict_to_load is not None:
            self.base_model.load_state_dict(state_dict_list[0])
        neighbor_weight_list = weight_list[1:]
        neighbor_flattened_params_list = [self._get_flattened_model_params(x.items()).to(self.device) for x in state_dict_list[1:]]
        self.train()
        with torch.enable_grad():
            for epoch_i in range(self.sync_every_n_epoch):
                num_samples = 0
                epoch_log = defaultdict(lambda : 0.0)
                for batch in self.train_dataloader:
                    x, y, x_attr, y_attr = batch
                    x = x.to(self.device) if (x is not None) else None
                    y = y.to(self.device) if (y is not None) else None
                    x_attr = x_attr.to(self.device) if (x_attr is not None) else None
                    y_attr = y_attr.to(self.device) if (y_attr is not None) else None
                    data = dict(
                        x=x, x_attr=x_attr, y=y, y_attr=y_attr
                    )
                    y_pred = self(data)
                    loss = nn.MSELoss()(y_pred, y)

                    # Laplacian regularization: \sum_{j!=i} a_{ij} * || W_j - W_i ||_2^2
                    mtl_loss = 0
                    self_flattend_params = self._get_flattened_model_params(self.base_model.named_parameters())
                    for weight_j, flattened_params_j in zip(neighbor_weight_list, neighbor_flattened_params_list):
                        mtl_loss += (weight_j * (self_flattend_params - flattened_params_j).norm('fro'))

                    loss = loss + self.mtl_lambda * mtl_loss
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    num_samples += x.shape[0]
                    metrics = unscaled_metrics(y_pred, y, self.feature_scaler, 'train')
                    epoch_log['train/loss'] += loss.detach() * x.shape[0]
                    if type(mtl_loss) is not int:
                        epoch_log['train/mtl_loss'] += mtl_loss.detach() * x.shape[0]
                    else:
                        epoch_log['train/mtl_loss'] += torch.zeros([]).float()
                    for k in metrics:
                        epoch_log[k] += metrics[k] * x.shape[0]
                    self.global_step += 1
                for k in epoch_log:
                    epoch_log[k] /= num_samples
                    epoch_log[k] = epoch_log[k].cpu()
        # self.cpu()
        state_dict = self.base_model.state_dict()
        for k in state_dict:
            state_dict[k] = state_dict[k].detach().to('cpu')
        epoch_log['num_samples'] = num_samples
        epoch_log['global_step'] = self.global_step
        epoch_log = dict(**epoch_log)

        return {
            'state_dict': state_dict, 'log': epoch_log
        }

    def local_eval(self, dataloader, name, state_dict_to_load):
        if state_dict_to_load is not None:
            self.base_model.load_state_dict(state_dict_to_load)
        self.eval()
        with torch.no_grad():
            num_samples = 0
            epoch_log = defaultdict(lambda : 0.0)
            for batch in dataloader:
                x, y, x_attr, y_attr = batch
                x = x.to(self.device) if (x is not None) else None
                y = y.to(self.device) if (y is not None) else None
                x_attr = x_attr.to(self.device) if (x_attr is not None) else None
                y_attr = y_attr.to(self.device) if (y_attr is not None) else None
                data = dict(
                    x=x, x_attr=x_attr, y=y, y_attr=y_attr
                )
                y_pred = self(data)
                loss = nn.MSELoss()(y_pred, y)
                num_samples += x.shape[0]
                metrics = unscaled_metrics(y_pred, y, self.feature_scaler, name)
                epoch_log['{}/loss'.format(name)] += loss.detach() * x.shape[0]
                for k in metrics:
                    epoch_log[k] += metrics[k] * x.shape[0]
            for k in epoch_log:
                epoch_log[k] /= num_samples
                epoch_log[k] = epoch_log[k].cpu()
        # self.cpu()
        epoch_log['num_samples'] = num_samples
        epoch_log = dict(**epoch_log)

        return {'log': epoch_log}

    def local_validation(self, state_dict_to_load):
        return self.local_eval(self.val_dataloader, 'val', state_dict_to_load)

    def local_test(self, state_dict_to_load):
        return self.local_eval(self.test_dataloader, 'test', state_dict_to_load)

    @staticmethod
    def client_local_execute(device, order, hparams_list):
        torch.cuda.empty_cache()
        if (type(device) is str) and (device.startswith('cuda:')):
            cuda_id = int(device.split(':')[1])
            # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            # os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_id)
            device = torch.device('cuda:{}'.format(cuda_id))
            # device = torch.device('cuda:0')
        elif type(device) is torch.device:
            pass
        else:
            device = torch.device('cpu')
        torch.cuda.set_device(device)
        res_list = []
        for hparams in hparams_list:
            state_dict_to_load = hparams['state_dict_to_load']
            client = FedMTLNodePredictorClient(client_device=device, **hparams)
            if order == 'train':
                res = client.local_train(state_dict_to_load)
            elif order == 'val':
                res = client.local_validation(state_dict_to_load)
            elif order == 'test':
                res = client.local_test(state_dict_to_load)
            else:
                del client
                torch.cuda.empty_cache()
                raise NotImplementedError()
            del client
            torch.cuda.empty_cache()
            res_list.append(res)
        return res_list

    # def cuda(self, device, manual_control=False):
    #     if not manual_control:
    #         return super().to('cpu')
    #     else: # only move to GPU when manual_control explicitly set!
    #         return super().cuda(device)

    # def to(self, device, manual_control=False, *args, **kwargs):
    #     if not manual_control:
    #         return super().to('cpu')
    #     else:
    #         return super().to(*args, **kwargs)


# def fednodepredictorclient_


class FedMTLNodePredictor(LightningModule):
    def __init__(self, hparams, identical_agg_model, *args, **kwargs):
        super().__init__()
        self.hparams = hparams
        self.identical_agg_model = identical_agg_model
        self.base_model = None
        self.setup(None)

    def forward(self, x):
        # return self.base_model(x)
        raise NotImplementedError()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--weight_decay', type=float, default=0.0)
        parser.add_argument('--mtl_lambda', type=float, default=0.0)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--sync_every_n_epoch', type=int, default=5)
        parser.add_argument('--mp_worker_num', type=int, default=8)
        return parser

    def prepare_data(self):
        pass

    def setup(self, step):
        # must avoid repeated init!!! otherwise loaded model weights will be re-initialized!!!
        if self.base_model is not None:
            return
        data = load_dataset(dataset_name=self.hparams.dataset)
        self.data = data
        # Each node (client) has its own model and optimizer
        # Assigning data, model and optimizer for each client
        num_clients = data['train']['x'].shape[2]
        # num_clients = 4
        input_size = self.data['train']['x'].shape[-1] + self.data['train']['x_attr'].shape[-1]
        output_size = self.data['train']['y'].shape[-1]
        client_params_list = []
        for client_i in range(num_clients):
            client_datasets = {}
            for name in ['train', 'val', 'test']:
                client_datasets[name] = TensorDataset(
                    data[name]['x'][:, :, client_i:client_i+1, :],
                    data[name]['y'][:, :, client_i:client_i+1, :],
                    data[name]['x_attr'][:, :, client_i:client_i+1, :],
                    data[name]['y_attr'][:, :, client_i:client_i+1, :]
                )
            client_params = {}
            client_params.update(
                optimizer_name='Adam',
                train_dataset=client_datasets['train'],
                val_dataset=client_datasets['val'],
                test_dataset=client_datasets['test'],
                feature_scaler=self.data['feature_scaler'],
                input_size=input_size,
                output_size=output_size,
                start_global_step=0,
                **self.hparams
            )
            client_params_list.append(client_params)
        self.client_params_list = client_params_list

        if self.identical_agg_model:
            self.base_model = getattr(base_models, self.hparams.base_model_name)(input_size=input_size, output_size=output_size, **self.hparams)
        else:
            self.base_model = []
            for idx in range(num_clients):
                self.base_model.append(
                    getattr(base_models, self.hparams.base_model_name)(input_size=input_size, output_size=output_size, **self.hparams)
                )
                self.base_model[-1].load_state_dict(deepcopy(self.base_model[0].state_dict()))
            self.base_model = nn.ModuleList(self.base_model)

        # preprocess edge_index, edge_attr to lists of neighbor weights
        edge_index, edge_attr = self.data['train']['edge_index'], self.data['train']['edge_attr']
        edge_index = edge_index.data.cpu().numpy()
        edge_attr = edge_attr.data.cpu().numpy()
        self.lists_of_neighbors = defaultdict(list)
        for eid, (s, t) in enumerate(edge_index.transpose()):
            if s != t:
                self.lists_of_neighbors[s].append((edge_attr[eid].item(), t))

    def _get_copied_ith_model(self, idx):
        if self.identical_agg_model:
            return deepcopy(self.base_model.state_dict())
        else:
            return deepcopy(self.base_model[idx].state_dict())

    def _get_weighted_neighboring_models(self, idx, copied_model_state_dicts):
        weighted_neighbors = [(None, copied_model_state_dicts[idx])]
        for n_w, n_id in self.lists_of_neighbors[idx]:
            weighted_neighbors.append((n_w, copied_model_state_dicts[n_id]))
        return weighted_neighbors

    def train_dataloader(self):
        # return a fake dataloader for running the loop
        return DataLoader([0,])

    def val_dataloader(self):
        return DataLoader([0,])

    def test_dataloader(self):
        return DataLoader([0,])

    def configure_optimizers(self):
        return None

    def backward(self, trainer, loss, optimizer, optimizer_idx):
        return None

    def training_step(self, batch, batch_idx):
        # 1. train locally and collect uploaded local train results
        local_train_results = []

        # copy all models
        copied_model_state_dicts = []
        for client_i, client_params in enumerate(self.client_params_list):
            copied_model_state_dicts.append(self._get_copied_ith_model(client_i))
        # fill neighbors
        for client_i, client_params in enumerate(self.client_params_list):
            client_params.update(state_dict_to_load=self._get_weighted_neighboring_models(client_i, copied_model_state_dicts))

        if self.hparams.mp_worker_num <= 1:
            for client_i, client_params in enumerate(self.client_params_list):
                local_train_result = FedMTLNodePredictorClient.client_local_execute(batch.device, 'train', **client_params)
                local_train_results.append(local_train_result)
        else:
            pool = mp.Pool(self.hparams.mp_worker_num)
            for worker_i, client_params in enumerate(np.array_split(self.client_params_list, self.hparams.mp_worker_num)):
                # gpu_list = list(map(str, os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
                gpu_list = list(range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))))
                device_name = 'cuda:{}'.format(gpu_list[worker_i % len(gpu_list)])
                local_train_results.append(pool.apply_async(FedMTLNodePredictorClient.client_local_execute, args=(
                    device_name, 'train', client_params)))
            pool.close()
            pool.join()
            local_train_results = list(map(lambda x: x.get(), local_train_results))
            local_train_results = list(itertools.chain.from_iterable(local_train_results))
        # update global steps for all clients
        for ltr, client_params in zip(local_train_results, self.client_params_list):
            client_params.update(start_global_step=ltr['log']['global_step'])
        # 2. aggregate
        agg_local_train_results = self.aggregate_local_train_results(local_train_results)
        # 3. update aggregated weights
        # if agg_local_train_results['state_dict'] is not None:
        if self.identical_agg_model:
            self.base_model.load_state_dict(agg_local_train_results['state_dict'])
        else:
            for idx in range(len(self.base_model)):
                self.base_model[idx].load_state_dict(agg_local_train_results['state_dict'][idx])
        agg_log = agg_local_train_results['log']
        # 3. send aggregated weights to all clients
        # if self.last_round_weight is not None:
        #     for client_i, client in enumerate(self.clients):
        #         client.load_weights(deepcopy(agg_state_dict))
        log = agg_log
        return {'loss': torch.tensor(0).float(), 'progress_bar': log, 'log': log}

    def aggregate_local_train_results(self, local_train_results):
        return {
            'state_dict': self.aggregate_local_train_state_dicts(
                [ltr['state_dict'] for ltr in local_train_results]
            ),
            'log': self.aggregate_local_logs(
                [ltr['log'] for ltr in local_train_results]
            )
        }

    def aggregate_local_train_state_dicts(self, local_train_state_dicts):
        raise NotImplementedError()

    def aggregate_local_logs(self, local_logs):
        agg_log = deepcopy(local_logs[0])
        for k in agg_log:
            agg_log[k] = 0
            for local_log in local_logs:
                if k == 'num_samples':
                    agg_log[k] += local_log[k]
                else:
                    agg_log[k] += local_log[k] * local_log['num_samples']
        for k in agg_log:
            if k != 'num_samples':
                agg_log[k] /= agg_log['num_samples']
        return agg_log

    def training_epoch_end(self, outputs):
        # already averaged!
        log = outputs[0]['log']
        log.pop('num_samples')
        return {'log': log, 'progress_bar': log}

    def validation_step(self, batch, batch_idx):
        # 1. vaidate locally and collect uploaded local train results
        local_val_results = []
        for client_i, client_params in enumerate(self.client_params_list):
            client_params.update(state_dict_to_load=self._get_copied_ith_model(client_i))
        if self.hparams.mp_worker_num <= 1:
            for client_i, client_params in enumerate(self.client_params_list):
                local_val_result = FedMTLNodePredictorClient.client_local_execute(batch.device, 'val', **client_params)
                local_val_results.append(local_val_result)
        else:
            pool = mp.Pool(self.hparams.mp_worker_num)
            for worker_i, client_params in enumerate(np.array_split(self.client_params_list, self.hparams.mp_worker_num)):
                # gpu_list = list(map(str, os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
                gpu_list = list(range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))))
                device_name = 'cuda:{}'.format(gpu_list[worker_i % len(gpu_list)])
                local_val_results.append(pool.apply_async(
                    FedMTLNodePredictorClient.client_local_execute, args=(
                        device_name, 'val', client_params)
                ))
            pool.close()
            pool.join()
            local_val_results = list(map(lambda x: x.get(), local_val_results))
            local_val_results = list(itertools.chain.from_iterable(local_val_results))
        # 2. aggregate
        log = self.aggregate_local_logs([x['log'] for x in local_val_results])
        return {'progress_bar': log, 'log': log}

    def validation_epoch_end(self, outputs):
        return self.training_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        # 1. vaidate locally and collect uploaded local train results
        local_val_results = []
        for client_i, client_params in enumerate(self.client_params_list):
            client_params.update(state_dict_to_load=self._get_copied_ith_model(client_i))
        if self.hparams.mp_worker_num <= 1:
            for client_i, client_params in enumerate(self.client_params_list):
                local_val_result = FedMTLNodePredictorClient.client_local_execute(batch.device, 'test', **client_params)
                local_val_results.append(local_val_result)
        else:
            pool = mp.Pool(self.hparams.mp_worker_num)
            for worker_i, client_params in enumerate(np.array_split(self.client_params_list, self.hparams.mp_worker_num)):
                # gpu_list = list(map(str, os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
                gpu_list = list(range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))))
                device_name = 'cuda:{}'.format(gpu_list[worker_i % len(gpu_list)])
                local_val_results.append(pool.apply_async(
                    FedMTLNodePredictorClient.client_local_execute, args=(
                        device_name, 'test', client_params)
                ))
            pool.close()
            pool.join()
            local_val_results = list(map(lambda x: x.get(), local_val_results))
            local_val_results = list(itertools.chain.from_iterable(local_val_results))
        # 2. aggregate
        log = self.aggregate_local_logs([x['log'] for x in local_val_results])
        return {'progress_bar': log, 'log': log}

    def test_epoch_end(self, outputs):
        return self.training_epoch_end(outputs)


class FixedGraphFedMTLNodePredictor(FedMTLNodePredictor):
    def __init__(self, hparams, *args, **kwargs):
        super().__init__(hparams, False, *args, **kwargs)

    def aggregate_local_train_state_dicts(self, local_train_state_dicts):
        return local_train_state_dicts
