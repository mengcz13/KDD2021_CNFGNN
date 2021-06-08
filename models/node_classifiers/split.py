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
from torch_geometric.nn import GCNConv

from datasets.datasets import load_dataset
import models.base_models as base_models


class SplitNodeClassifierClient(nn.Module):
    def __init__(self, enc_base_model_name, dec_base_model_name,
        optimizer_name,
        input_size, hidden_size, output_size,
        data,
        train_mask, val_mask, test_mask, 
        lr, batch_size,
        *args, **kwargs):
        super().__init__()
        self.enc_base_model_name = enc_base_model_name
        self.dec_base_model_name = dec_base_model_name
        self.optimizer_name = optimizer_name
        self.data = data
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.lr = lr
        self.batch_size = batch_size

        self.enc_base_model_class = getattr(base_models, self.enc_base_model_name)
        # self.enc_base_model = self.enc_base_model_class(
        #     input_size=input_size, hidden_size=hidden_size, output_size=hidden_size,
        #     final_activation=True, **kwargs)
        self.enc_base_model = self.enc_base_model_class(
            input_size=input_size, hidden_size=hidden_size, output_size=hidden_size,
            final_activation=False, **kwargs)
        self.dec_base_model_class = getattr(base_models, self.dec_base_model_name)
        self.dec_base_model = self.dec_base_model_class(
            input_size=hidden_size, hidden_size=hidden_size, output_size=output_size,
            final_activation=False, **kwargs)
        self.enc_optimizer = getattr(torch.optim, self.optimizer_name)(self.enc_base_model.parameters(), lr=self.lr)
        self.dec_optimizer = getattr(torch.optim, self.optimizer_name)(self.dec_base_model.parameters(), lr=self.lr)

    def forward(self, x, stage):
        x = x.to(next(self.parameters()).device)
        if stage == 'enc':
            return self.enc_base_model(x)
        elif stage == 'dec':
            return self.dec_base_model(x)
        else:
            raise NotImplementedError()

    def local_encode_forward(self):
        self.encoding = self(x=self.data, stage='enc')
        return self.encoding

    def local_decode_forward(self, input_hidden):
        self.input_hidden = input_hidden.clone().detach().requires_grad_(True)
        self.input_hidden.retain_grad()
        self.m_out = self(x=Data(x=self.input_hidden), stage='dec')
        # decoded output is not returned to the server!

    def local_backward(self, grads=None, stage='dec'):
        if stage == 'dec':
            loss = nn.CrossEntropyLoss()(self.m_out, self.data.y) * float(self.train_mask)
            loss.backward()
            y_pred = self.m_out.argmax(dim=1)
            accu = (y_pred == self.data.y).float().mean()
            num_samples = self.data.y.shape[0]
            epoch_log = {
                'train/loss': loss.detach(),
                'train/accu': accu.detach(),
                'num_samples': num_samples
            }
            return {
                'log': epoch_log, 'grad': self.input_hidden.grad
            }
        elif stage == 'enc':
            self.encoding.backward(grads)
        else:
            raise NotImplementedError()

    def local_optimizer_step(self):
        self.enc_optimizer.step()
        self.dec_optimizer.step()

    def local_optimizer_zero_grad(self):
        self.enc_optimizer.zero_grad()
        self.dec_optimizer.zero_grad()

    def local_eval(self, name):
        loss = nn.CrossEntropyLoss()(self.m_out, self.data.y)
        y_pred = self.m_out.argmax(dim=1)
        accu = (y_pred == self.data.y).float().mean()
        num_samples = self.data.y.shape[0]
        epoch_log = {
            '{}/loss'.format(name): loss.detach(),
            '{}/accu'.format(name): accu.detach(),
            'num_samples': num_samples
        }
        state_dict = self.state_dict()
        return {
            'log': epoch_log
        }

    def local_validation(self):
        return self.local_eval('val')

    def local_test(self):
        return self.local_eval('test')


class SplitGCNNodeClassifier(LightningModule):
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
        # parser.add_argument('--sync_every_n_epoch', type=int, default=5)
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
            client_data = Data(
                x=data.x[None, client_i, :], y=data.y[None, client_i]
            )
            client = SplitNodeClassifierClient(
                enc_base_model_name=self.hparams.base_model_name,
                dec_base_model_name=self.hparams.base_model_name,
                optimizer_name='Adam',
                input_size=data.x.shape[-1],
                output_size=data.y.unique().shape[0],
                data=client_data,
                train_mask=data.train_mask[client_i], 
                val_mask=data.val_mask[client_i],
                test_mask=data.test_mask[client_i],
                **self.hparams
            )
            self.clients.append(client)
        self.clients = nn.ModuleList(self.clients)

        self.prop_model = nn.Sequential(
            base_models.GCN(
                input_size=self.hparams.hidden_size,
                hidden_size=self.hparams.hidden_size,
                output_size=self.hparams.hidden_size,
                dropout=self.hparams.dropout
            ),
            nn.ReLU(),
            nn.Dropout(p=self.hparams.dropout)
        )
        self.prop_model_optimizer = torch.optim.Adam(self.prop_model.parameters(), lr=self.hparams.lr)

    def train_dataloader(self):
        # return a fake dataloader for running the loop
        return DataLoader([self.data,])

    def val_dataloader(self):
        return DataLoader([self.data,])

    def test_dataloader(self):
        return DataLoader([self.data,])

    def configure_optimizers(self):
        return None

    def backward(self, trainer, loss, optimizer, optimizer_idx):
        return None

    def training_step(self, batch, batch_idx):
        # 1. run encoding on all clients and collect embeddings
        encodings = []
        for client_i, client in enumerate(self.clients):
            encodings.append(client.local_encode_forward())
        encodings = torch.cat(encodings, dim=0) # N x F
        # 2. run propagation with the known graph structure and collected embeddings
        hiddens = self.prop_model(Data(x=encodings, edge_index=batch.edge_index)) # N x F
        # 3. run decoding on all clients
        for client_i, client in enumerate(self.clients):
            client.local_decode_forward(hiddens[None, client_i, :])
        # 4. run zero_grad for all optimizers
        self.prop_model.zero_grad()
        for client_i, client in enumerate(self.clients):
            client.local_optimizer_zero_grad()
        # 4. run backward on all clients
        local_train_logs = []
        hiddens_msg_grad = []
        for client_i, client in enumerate(self.clients):
            if client.train_mask:
                local_train_result = client.local_backward(stage='dec')
                local_train_logs.append(local_train_result['log'])
                hiddens_msg_grad.append(local_train_result['grad'])
            else:
                hiddens_msg_grad.append(hiddens.new_zeros(1, hiddens.shape[-1]))
        # print(hiddens_msg_grad[:5])
        hiddens_msg_grad = torch.cat(hiddens_msg_grad, dim=0)
        hiddens.backward(hiddens_msg_grad)
        # 5. run optimizers on the server and all clients
        self.prop_model_optimizer.step()
        for client_i, client in enumerate(self.clients):
            client.local_optimizer_step()
        # 6. aggregate local train results
        agg_state_dict = self.aggregate_local_train_state_dicts(
            [client.state_dict() for client in self.clients]
        )
        agg_log = self.aggregate_local_logs(local_train_logs)
        # for now always sync enc/dec
        for client_i, client in enumerate(self.clients):
            client.load_state_dict(deepcopy(agg_state_dict))
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
        agg_state_dict = {}
        for k in local_train_state_dicts[0]:
            agg_state_dict[k] = 0
            for ltsd in local_train_state_dicts:
                agg_state_dict[k] += ltsd[k]
            agg_state_dict[k] /= len(local_train_state_dicts)
        return agg_state_dict

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
        # 1. run encoding on all clients and collect embeddings
        encodings = []
        for client_i, client in enumerate(self.clients):
            encodings.append(client.local_encode_forward())
        encodings = torch.cat(encodings, dim=0) # N x F
        # 2. run propagation with the known graph structure and collected embeddings
        hiddens = self.prop_model(Data(x=encodings, edge_index=batch.edge_index)) # N x F
        # 3. run decoding on all clients
        for client_i, client in enumerate(self.clients):
            client.local_decode_forward(hiddens[None, client_i, :])
        # 4. run backward on all clients
        local_val_results = []
        for client_i, client in enumerate(self.clients):
            if client.val_mask:
                local_val_results.append(client.local_validation())
        # 6. aggregate local val results
        log = self.aggregate_local_logs([x['log'] for x in local_val_results])
        return {'progress_bar': log, 'log': log}

    def validation_epoch_end(self, outputs):
        return self.training_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        # 1. run encoding on all clients and collect embeddings
        encodings = []
        for client_i, client in enumerate(self.clients):
            encodings.append(client.local_encode_forward())
        encodings = torch.cat(encodings, dim=0) # N x F
        # if required, save node encodings
        if self.hparams.save_node_encodings_test:
            # self.test_encodings = encodings.data.cpu().numpy()
            self.test_encodings = {
                'input': self.data.x.data.cpu().numpy(),
                'encodings': encodings.data.cpu().numpy(),
                'train_mask': self.data.train_mask.data.cpu().numpy(),
                'val_mask': self.data.val_mask.cpu().numpy(),
                'test_mask': self.data.test_mask.cpu().numpy()
            }
        # 2. run propagation with the known graph structure and collected embeddings
        hiddens = self.prop_model(Data(x=encodings, edge_index=batch.edge_index)) # N x F
        # 3. run decoding on all clients
        for client_i, client in enumerate(self.clients):
            client.local_decode_forward(hiddens[None, client_i, :])
        # 4. run backward on all clients
        local_test_results = []
        for client_i, client in enumerate(self.clients):
            if client.test_mask:
                local_test_results.append(client.local_test())
        # 6. aggregate local test results
        log = self.aggregate_local_logs([x['log'] for x in local_test_results])
        return {'progress_bar': log, 'log': log}

    def test_epoch_end(self, outputs):
        return self.training_epoch_end(outputs)

