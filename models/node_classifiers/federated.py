from argparse import ArgumentParser
from multiprocessing import cpu_count
from copy import deepcopy
from collections import defaultdict

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torch_geometric.data import DataLoader, Data
from torch.utils.data import TensorDataset

from datasets.datasets import load_dataset
import models.base_models as base_models


class FedNodeClassifierClient(nn.Module):
    def __init__(self, base_model_name, optimizer_name,
        train_dataset, val_dataset, test_dataset, 
        train_mask, val_mask, test_mask, 
        sync_every_n_epoch, lr, weight_decay, batch_size,
        *args, **kwargs):
        super().__init__()
        self.base_model_name = base_model_name
        self.optimizer_name = optimizer_name
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.sync_every_n_epoch = sync_every_n_epoch
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size

        self.base_model_class = getattr(base_models, self.base_model_name)
        self.base_model = self.base_model_class(**kwargs)
        self.optimizer = getattr(torch.optim, self.optimizer_name)(self.base_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        if self.val_dataset:
            self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            self.val_dataloader = self.train_dataloader
        if self.test_dataset:
            self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            self.test_dataloader = self.train_dataloader

    def forward(self, x):
        return self.base_model(x)

    def local_train(self, device):
        # self.to(device)
        self.train()
        with torch.enable_grad():
            for epoch_i in range(self.sync_every_n_epoch):
                num_samples = 0
                epoch_log = defaultdict(lambda : 0.0)
                for data in self.train_dataloader:
                    data = data.to(device)
                    x, y = data.x, data.y
                    m_out = self(data)
                    loss = nn.CrossEntropyLoss()(m_out, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    y_pred = m_out.argmax(dim=1)
                    accu = (y_pred == y).float().mean()
                    num_samples += x.shape[0]
                    epoch_log['train/loss'] += loss.detach() * x.shape[0]
                    epoch_log['train/accu'] += accu.detach() * x.shape[0]
                for k in epoch_log:
                    epoch_log[k] /= num_samples
        # self.cpu()
        state_dict = self.base_model.state_dict()
        epoch_log['num_samples'] = num_samples
        return {
            'state_dict': state_dict, 'log': epoch_log
        }

    def local_eval(self, dataloader, device, name):
        # self.to(device)
        self.eval()
        with torch.no_grad():
            num_samples = 0
            epoch_log = defaultdict(lambda : 0.0)
            for data in dataloader:
                data = data.to(device)
                x, y = data.x, data.y
                m_out = self(data)
                loss = nn.CrossEntropyLoss()(m_out, y)
                y_pred = m_out.argmax(dim=1)
                accu = (y_pred == y).float().mean()
                num_samples += x.shape[0]
                epoch_log['{}/loss'.format(name)] += loss.detach() * x.shape[0]
                epoch_log['{}/accu'.format(name)] += accu.detach() * x.shape[0]
            for k in epoch_log:
                epoch_log[k] /= num_samples
        # self.cpu()
        epoch_log['num_samples'] = num_samples
        return {'log': epoch_log}

    def local_validation(self, device):
        return self.local_eval(self.val_dataloader, device, 'val')

    def local_test(self, device):
        return self.local_eval(self.test_dataloader, device, 'test')

    def load_weights(self, state_dict):
        self.base_model.load_state_dict(state_dict)
        return self

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


class FedNodeClassifier(LightningModule):
    def __init__(self, hparams, *args, **kwargs):
        super().__init__()
        self.hparams = hparams
        self.clients = None
        self.setup(None)

    def forward(self, x):
        # return self.base_model(x)
        raise NotImplemented()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--weight_decay', type=float, default=0.0)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--sync_every_n_epoch', type=int, default=5)
        return parser

    def prepare_data(self):
        data = load_dataset(dataset_name=self.hparams.dataset, split_edges=False)

    def setup(self, step):
        if self.clients is not None:
            return
        data = load_dataset(dataset_name=self.hparams.dataset, split_edges=False)
        self.data = data
        # Each node (client) has its own model and optimizer
        # Assigning data, model and optimizer for each client
        self.clients = []
        num_clients = data.x.shape[0]
        for client_i in range(num_clients):
            train_dataset = [Data(
                x=data.x[None, client_i, :], y=data.y[None, client_i]
            ),]
            client = FedNodeClassifierClient(
                optimizer_name='Adam',
                train_dataset=train_dataset, val_dataset=None, test_dataset=None,
                train_mask=data.train_mask[client_i], 
                val_mask=data.val_mask[client_i],
                test_mask=data.test_mask[client_i],
                input_size=data.x.shape[-1],
                output_size=data.y.unique().shape[0],
                **self.hparams
            )
            self.clients.append(client)
            
        # test FedAvg: assign 10 clients, each with 10% training/val/test data, all clients participate in train/val/test
        # train_ids = torch.where(data.train_mask)[0]
        # tidl = train_ids.shape[0] // 10
        # val_ids = torch.where(data.val_mask)[0]
        # vidl = val_ids.shape[0] // 10
        # test_ids = torch.where(data.test_mask)[0]
        # teidl = test_ids.shape[0] // 10
        # for client_i in range(21):
        #     if client_i < 10:
        #         train_dataset = [Data(
        #             x=data.x[train_ids[tidl*client_i:tidl*(client_i+1)], :],
        #             y=data.y[train_ids[tidl*client_i:tidl*(client_i+1)]]
        #         ),]
        #         client = FedNodeClassifierClient(
        #             optimizer_name='Adam',
        #             train_dataset=train_dataset, val_dataset=None, test_dataset=None,
        #             # train_mask=data.train_mask[client_i], 
        #             # val_mask=data.val_mask[client_i],
        #             # test_mask=data.test_mask[client_i],
        #             train_mask=True, val_mask=False, test_mask=False,
        #             input_size=data.x.shape[-1],
        #             output_size=data.y.unique().shape[0],
        #             **self.hparams
        #         )
        #     elif client_i < 20:
        #         client_i = client_i - 10
        #         train_dataset = [Data(
        #             x=data.x[val_ids[vidl*client_i:vidl*(client_i+1)], :],
        #             y=data.y[val_ids[vidl*client_i:vidl*(client_i+1)]]
        #         ),]
        #         print(train_dataset[0])
        #         client = FedNodeClassifierClient(
        #             optimizer_name='Adam',
        #             train_dataset=train_dataset, val_dataset=None, test_dataset=None,
        #             # train_mask=data.train_mask[client_i], 
        #             # val_mask=data.val_mask[client_i],
        #             # test_mask=data.test_mask[client_i],
        #             train_mask=False, val_mask=True, test_mask=False,
        #             input_size=data.x.shape[-1],
        #             output_size=data.y.unique().shape[0],
        #             **self.hparams
        #         )
        #     else:
        #         client_i = client_i - 20
        #         train_dataset = [Data(
        #             x=data.x[test_ids, :],
        #             y=data.y[test_ids]
        #         ),]
        #         print(train_dataset[0])
        #         client = FedNodeClassifierClient(
        #             optimizer_name='Adam',
        #             train_dataset=train_dataset, val_dataset=None, test_dataset=None,
        #             # train_mask=data.train_mask[client_i], 
        #             # val_mask=data.val_mask[client_i],
        #             # test_mask=data.test_mask[client_i],
        #             train_mask=False, val_mask=False, test_mask=True,
        #             input_size=data.x.shape[-1],
        #             output_size=data.y.unique().shape[0],
        #             **self.hparams
        #         )
        #     self.clients.append(client)
        self.clients = nn.ModuleList(self.clients)

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
        for client_i, client in enumerate(self.clients):
            if client.train_mask:
                local_train_result = client.local_train(batch.device)
                local_train_results.append(local_train_result)
        # 2. aggregate
        agg_local_train_results = self.aggregate_local_train_results(local_train_results)
        agg_state_dict = agg_local_train_results['state_dict']
        agg_log = agg_local_train_results['log']
        # 3. send aggregated weights to all clients
        for client_i, client in enumerate(self.clients):
            client.load_weights(deepcopy(agg_state_dict))
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
        for client_i, client in enumerate(self.clients):
            if client.val_mask:
                local_val_result = client.local_validation(batch.device)
                local_val_results.append(local_val_result)
        # 2. aggregate
        log = self.aggregate_local_logs([x['log'] for x in local_val_results])
        return {'progress_bar': log, 'log': log}

    def validation_epoch_end(self, outputs):
        return self.training_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        # 1. vaidate locally and collect uploaded local train results
        local_val_results = []
        for client_i, client in enumerate(self.clients):
            if client.test_mask:
                local_val_result = client.local_test(batch.device)
                local_val_results.append(local_val_result)
        # 2. aggregate
        log = self.aggregate_local_logs([x['log'] for x in local_val_results])
        return {'progress_bar': log, 'log': log}

    def test_epoch_end(self, outputs):
        return self.training_epoch_end(outputs)


class FedAvgNodeClassifier(FedNodeClassifier):
    def __init__(self, hparams, *args, **kwargs):
        super().__init__(hparams, *args, **kwargs)

    def aggregate_local_train_state_dicts(self, local_train_state_dicts):
        agg_state_dict = {}
        for k in local_train_state_dicts[0]:
            agg_state_dict[k] = 0
            for ltsd in local_train_state_dicts:
                agg_state_dict[k] += ltsd[k]
            agg_state_dict[k] /= len(local_train_state_dicts)
        return agg_state_dict