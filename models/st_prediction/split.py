from argparse import ArgumentParser
from multiprocessing import cpu_count
from copy import deepcopy
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_adj
from torch.utils.data import TensorDataset, Dataset, DataLoader

from datasets.st_datasets import load_dataset
import models.base_models as base_models
from models.st_prediction.standalone import unscaled_metrics


class EdgeDataset(Dataset):
    def __init__(self, edge_index, edge_attr):
        self.edge_index = edge_index
        self.edge_attr = edge_attr

    def __len__(self):
        return len(self.edge_index)

    def __getitem__(self, index):
        return self.edge_index[index], self.edge_attr[index]

    @staticmethod
    def collate_fn(batch):
        return batch


class Conv1DLayer(nn.Module):
    def __init__(self, input_size, output_size, final_activation=False):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=input_size, out_channels=output_size, kernel_size=1
        )
        self.final_activation = final_activation

    def forward(self, x):
        batch_num, ts, n, fdim = x.shape
        x = x.permute(0, 2, 3, 1).flatten(0, 1)
        out = self.conv(x)
        if self.final_activation:
            out = nn.functional.relu(out)
        out = out.view(batch_num, n, -1, ts).permute(0, 3, 1, 2)
        return out


class SplitNodePredictorClient(nn.Module):
    def __init__(self, enc_base_model_name, dec_base_model_name,
        optimizer_name,
        input_size, hidden_size, output_size,
        train_dataset, val_dataset, test_dataset, feature_scaler,
        lr, weight_decay, batch_size,
        *args, **kwargs):
        super().__init__()
        self.enc_base_model_name = enc_base_model_name
        self.dec_base_model_name = dec_base_model_name
        self.optimizer_name = optimizer_name
        self.train_dataset = train_dataset
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        self.train_dataloader_iter = iter(self.train_dataloader)
        self.val_dataset = val_dataset
        self.val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.val_dataloader_iter = iter(self.val_dataloader)
        self.test_dataset = test_dataset
        self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        self.test_dataloader_iter = iter(self.test_dataloader)
        self.feature_scaler = feature_scaler
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size

        # self.enc_base_model_class = getattr(base_models, self.enc_base_model_name)
        # self.enc_base_model = self.enc_base_model_class(
        #     input_size=input_size, hidden_size=hidden_size, output_size=hidden_size,
        #     final_activation=False, **kwargs)
        # self.dec_base_model_class = getattr(base_models, self.dec_base_model_name)
        # self.dec_base_model = self.dec_base_model_class(
        #     input_size=hidden_size, hidden_size=hidden_size, output_size=output_size,
        #     final_activation=False, **kwargs)
        self.enc_base_model = Conv1DLayer(input_size, hidden_size, final_activation=False)
        self.dec_base_model = Conv1DLayer(hidden_size, output_size, final_activation=False)
        self.enc_optimizer = getattr(torch.optim, self.optimizer_name)(self.enc_base_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.dec_optimizer = getattr(torch.optim, self.optimizer_name)(self.dec_base_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def forward(self, x, stage):
        x = x.to(next(self.parameters()).device)
        if stage == 'enc':
            return self.enc_base_model(x)
        elif stage == 'dec':
            return self.dec_base_model(x)
        else:
            raise NotImplementedError()

    def local_encode_forward(self, dataset):
        if dataset == 'train':
            try:
                self.data = next(self.train_dataloader_iter) # list of tensors
            except StopIteration:
                self.train_dataloader_iter = iter(self.train_dataloader)
                self.data = next(self.train_dataloader_iter)
        elif dataset == 'val':
            try:
                self.data = next(self.val_dataloader_iter) # list of tensors
            except StopIteration:
                self.val_dataloader_iter = iter(self.val_dataloader)
                self.data = next(self.val_dataloader_iter)
        elif dataset == 'test':
            try:
                self.data = next(self.test_dataloader_iter) # list of tensors
            except StopIteration:
                self.test_dataloader_iter = iter(self.test_dataloader)
                self.data = next(self.test_dataloader_iter)
        x, y, x_attr, y_attr = self.data
        self.y = y.to(next(self.parameters()).device)
        x_enc = self(torch.cat((x, x_attr), dim=-1), stage='enc')
        x_attr_enc = x_enc.new_zeros(*x_enc.shape[:-1], 0) # for compatibility
        y_enc = self(torch.cat((y, y_attr), dim=-1), stage='enc')
        y_attr_enc = y_enc.new_zeros(*y_enc.shape[:-1], 0) # for compatibility
        self.encoding = {
            'x': x_enc, 'x_attr': x_attr_enc,
            'y': y_enc, 'y_attr': y_attr_enc
        }
        return self.encoding

    def local_decode_forward(self, input_hidden):
        self.input_hidden = input_hidden.clone().detach().requires_grad_(True)
        self.input_hidden.retain_grad()
        self.y_pred = self(x=self.input_hidden, stage='dec')
        # decoded output is not returned to the server!

    def local_backward(self, grads=None, stage='dec'):
        if stage == 'dec':
            loss = nn.MSELoss()(self.y_pred, self.y)
            loss.backward()
            num_samples = self.y.shape[0]
            epoch_log = {
                'train/loss': loss.detach(),
                'num_samples': num_samples
            }
            epoch_log.update(**unscaled_metrics(self.y_pred, self.y, self.feature_scaler, 'train'))
            return {
                'log': epoch_log, 'grad': self.input_hidden.grad
            }
        elif stage == 'enc':
            # self.encoding.backward(grads)
            # raise NotImplementedError()
            for name in ['x', 'y']:
                if grads[name] is not None:
                    self.encoding[name].backward(grads[name])
        else:
            raise NotImplementedError()

    def local_optimizer_step(self):
        self.enc_optimizer.step()
        self.dec_optimizer.step()

    def local_optimizer_zero_grad(self):
        self.enc_optimizer.zero_grad()
        self.dec_optimizer.zero_grad()

    def local_eval(self, name):
        loss = nn.MSELoss()(self.y_pred, self.y)
        num_samples = self.y.shape[0]
        epoch_log = {
            '{}/loss'.format(name): loss.detach(),
            'num_samples': num_samples
        }
        epoch_log.update(**unscaled_metrics(self.y_pred, self.y, self.feature_scaler, name))
        return {
            'log': epoch_log
        }

    def local_validation(self):
        return self.local_eval('val')

    def local_test(self):
        return self.local_eval('test')

    @staticmethod
    def client_local_execute(clients, device, order, params):
        if (type(device) is str) and (device.startswith('cuda:')):
            cuda_id = int(device.split(':')[1])
            device = torch.device('cuda:{}'.format(cuda_id))
        elif type(device) is torch.device:
            pass
        else:
            device = torch.device('cpu')
        torch.cuda.set_device(device)
        res_list = []
        for client in clients:
            client.to(device)
            res_list.append(getattr(client, order)(**params))
        return res_list


class SplitGCNNodePredictor(LightningModule):
    def __init__(self, hparams, *args, **kwargs):
        super().__init__()
        self.hparams = hparams
        self.clients = None
        self.setup(None)
        self.train_started = False

    def forward(self, x):
        # return self.base_model(x)
        raise NotImplementedError()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--weight_decay', type=float, default=0.0)
        parser.add_argument('--batch_size', type=int, default=128)
        # parser.add_argument('--sync_every_n_epoch', type=int, default=5)
        parser.add_argument('--hetero_graph', action='store_true')
        return parser

    def prepare_data(self):
        pass

    def setup(self, step):
        if self.clients is not None:
            return
        data = load_dataset(dataset_name=self.hparams.dataset)
        self.data = data
        # Each node (client) has its own model and optimizer
        # Assigning data, model and optimizer for each client
        self.clients = []
        num_clients = data['train']['x'].shape[2]
        input_size = self.data['train']['x'].shape[-1] + self.data['train']['x_attr'].shape[-1]
        output_size = self.data['train']['y'].shape[-1]
        
        # shuffle trainset in advance with a fixed seed (42)
        train_sample_num = data['train']['x'].shape[0]
        shufflerng = np.random.RandomState(42)
        shuffled_train_idx = shufflerng.permutation(train_sample_num).astype(int)
        for name in ['x', 'y', 'x_attr', 'y_attr']:
            data['train'][name] = (data['train'][name])[shuffled_train_idx]
        if self.hparams.hetero_graph:
            for name in ['edge_index', 'edge_attr']:
                data['train'][name] = [data['train'][name][tid] for tid in shuffled_train_idx]

        self.graph_dataset = {}
        for name in ['train', 'val', 'test']:
            if self.hparams.hetero_graph:
                self.graph_dataset[name] = EdgeDataset(self.data[name]['edge_index'], self.data[name]['edge_attr'])
            else:
                self.graph_dataset[name] = TensorDataset(torch.zeros_like(self.data[name]['y'])) # fake dataset, used for 1 batch forward only
            
        for client_i in range(num_clients):
            client_datasets = {}
            for name in ['train', 'val', 'test']:
                client_datasets[name] = TensorDataset(
                    data[name]['x'][:, :, client_i:client_i+1, :],
                    data[name]['y'][:, :, client_i:client_i+1, :],
                    data[name]['x_attr'][:, :, client_i:client_i+1, :],
                    data[name]['y_attr'][:, :, client_i:client_i+1, :]
                )
            client = SplitNodePredictorClient(
                enc_base_model_name=self.hparams.base_model_name,
                dec_base_model_name=self.hparams.base_model_name,
                optimizer_name='Adam',
                input_size=input_size,
                output_size=output_size,
                train_dataset=client_datasets['train'],
                val_dataset=client_datasets['val'],
                test_dataset=client_datasets['test'],
                feature_scaler=self.data['feature_scaler'],
                **self.hparams
            )
            self.clients.append(client)
        self.clients = nn.ModuleList(self.clients)

        if self.hparams.prop_model_name == 'NRIDecoder':
            self.prop_model = base_models.NRIDecoder(
                input_size=self.hparams.hidden_size,
                hidden_size=self.hparams.hidden_size,
                output_size=self.hparams.hidden_size,
                dropout=self.hparams.dropout, cl_decay_steps=self.hparams.cl_decay_steps,
                use_curriculum_learning=self.hparams.use_curriculum_learning,
                decoder_unit_type=self.hparams.decoder_unit_type
            )
        elif self.hparams.prop_model_name == 'DCRNNModel':
            self.prop_model = base_models.DCRNNModel(
                adj_mx=to_dense_adj(self.data['train']['edge_index'], edge_attr=self.data['train']['edge_attr']).data.cpu().numpy()[0],
                num_graph_nodes=self.data['train']['x'].shape[2],
                input_dim=self.hparams.hidden_size,
                output_dim=self.hparams.hidden_size,
                seq_len=self.data['train']['x'].shape[1],
                horizon=self.data['train']['y'].shape[1],
                **self.hparams
            )
        self.prop_model_optimizer = torch.optim.Adam(self.prop_model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        # init layer params of DCRNN
        if self.hparams.prop_model_name == 'DCRNNModel':
            temp_dataloader = DataLoader(
                self.graph_dataset['val'],
                batch_size=1,
                collate_fn=EdgeDataset.collate_fn
            )
            batch = next(iter(temp_dataloader))
            self.validation_step(batch, None)

    def train_dataloader(self):
        # return a fake dataloader for running the loop
        return DataLoader(self.graph_dataset['train'], batch_size=self.hparams.batch_size, collate_fn=EdgeDataset.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.graph_dataset['val'], batch_size=self.hparams.batch_size, collate_fn=EdgeDataset.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.graph_dataset['test'], batch_size=self.hparams.batch_size, collate_fn=EdgeDataset.collate_fn)

    def configure_optimizers(self):
        return None

    def backward(self, trainer, loss, optimizer, optimizer_idx):
        return None

    def training_step(self, batch, batch_idx):
        if self.train_started is False:
            # reset val_dataloader on clients
            for client_i, client in enumerate(self.clients):
                client.val_dataloader_iter = iter(client.val_dataloader)
            self.train_started = True
        if self.hparams.hetero_graph:
            edge_index_batch, edge_attr_batch = list(zip(*batch))
        # 1. run encoding on all clients and collect embeddings
        encodings = [] # list of dicts of x, x_attr, y, y_attr (B x T x F)
        for client_i, client in enumerate(self.clients):
            encodings.append(client.local_encode_forward('train'))
        stacked_encodings = {}
        for k in encodings[0]:
            stacked_encodings[k] = torch.cat([enc[k] for enc in encodings], dim=2) # B x T x N x F
        if self.hparams.hetero_graph:
            to_batch_graph_list = []
            for bi in range(stacked_encodings['x'].shape[0]):
                to_batch_graph_list.append(Data(
                    x=stacked_encodings['x'][bi].permute(1, 0, 2), y=stacked_encodings['y'][bi].permute(1, 0, 2),
                    x_attr=stacked_encodings['x_attr'][bi].permute(1, 0, 2), y_attr=stacked_encodings['y_attr'][bi].permute(1, 0, 2),
                    edge_attr=edge_attr_batch[bi], edge_index=edge_index_batch[bi]))
            data = Batch.from_data_list(to_batch_graph_list)
        else:
            data = {'x': stacked_encodings['x'], 'y': stacked_encodings['y'], 
                'x_attr': stacked_encodings['x_attr'], 'y_attr': stacked_encodings['y_attr']}

        # keep gradients
        for name in ['x', 'y']: # 'x_attr' and 'y_attr' are zero vectors and not used
            data[name] = data[name].clone().detach().requires_grad_(True)
            data[name].retain_grad()

        # 2. run propagation with the known graph structure and collected embeddings
        hiddens = self.prop_model(data, batches_seen=self.global_step) # always B x T x N x F
        if self.hparams.hetero_graph:
            hiddens = hiddens.view(len(to_batch_graph_list), -1, hiddens.shape[1], hiddens.shape[2]).permute(0, 2, 1, 3) # B x T x N x F
        # 3. run decoding on all clients
        for client_i, client in enumerate(self.clients):
            client.local_decode_forward(hiddens[:, :, client_i:client_i+1, :])
        # 4. run zero_grad for all optimizers
        self.prop_model.zero_grad()
        for client_i, client in enumerate(self.clients):
            client.local_optimizer_zero_grad()
        # 4. run backward on all clients
        local_train_logs = []
        hiddens_msg_grad = []
        for client_i, client in enumerate(self.clients):
            local_train_result = client.local_backward(stage='dec')
            local_train_logs.append(local_train_result['log'])
            hiddens_msg_grad.append(local_train_result['grad'])
        hiddens_msg_grad = torch.cat(hiddens_msg_grad, dim=2)
        # 4.1 run backward on server graph model
        hiddens.backward(hiddens_msg_grad)
        # 4.2 collect grads on data and run backward on clients
        if self.hparams.hetero_graph:
            data_grads_all = {}
            for name in ['x', 'y']:
                if data[name].grad is not None:
                    data_grads_all[name] = data[name].grad.view(-1, len(self.clients), data[name].shape[1], data[name].shape[2]).permute(0, 2, 1, 3) # B x T x N x F
                else:
                    data_grads_all[name] = None
            for client_i, client in enumerate(self.clients):
                data_grads = {}
                for name in ['x', 'y']:
                    if data_grads_all[name] is not None:
                        data_grads[name] = data_grads_all[name][:, :, client_i:client_i+1, :]
                    else:
                        data_grads[name] = None
                client.local_backward(grads=data_grads, stage='enc')
        else:
            for client_i, client in enumerate(self.clients):
                data_grads = {}
                for name in ['x', 'y']:
                    if data[name].grad is not None:
                        data_grads[name] = data[name].grad[:, :, client_i:client_i+1, :] # B x T x 1 x F
                    else:
                        data_grads[name] = None
                client.local_backward(grads=data_grads, stage='enc')
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
        if self.hparams.hetero_graph:
            edge_index_batch, edge_attr_batch = list(zip(*batch))
        # 1. run encoding on all clients and collect embeddings
        encodings = [] # list of dicts of x, x_attr, y, y_attr (B x T x 1 x F)
        for client_i, client in enumerate(self.clients):
            encodings.append(client.local_encode_forward('val'))
        stacked_encodings = {}
        for k in encodings[0]:
            stacked_encodings[k] = torch.cat([enc[k] for enc in encodings], dim=2) # B x T x N x F
        # print(len(batch), len(edge_index_batch), len(stacked_encodings['x']))
        if self.hparams.hetero_graph:
            to_batch_graph_list = []
            for bi in range(stacked_encodings['x'].shape[0]):
                to_batch_graph_list.append(Data(
                    x=stacked_encodings['x'][bi].permute(1, 0, 2), y=stacked_encodings['y'][bi].permute(1, 0, 2),
                    x_attr=stacked_encodings['x_attr'][bi].permute(1, 0, 2), y_attr=stacked_encodings['y_attr'][bi].permute(1, 0, 2),
                    edge_attr=edge_attr_batch[bi], edge_index=edge_index_batch[bi]))
            data = Batch.from_data_list(to_batch_graph_list)
        else:
            data = {'x': stacked_encodings['x'], 'y': stacked_encodings['y'], 
                'x_attr': stacked_encodings['x_attr'], 'y_attr': stacked_encodings['y_attr']}

        # 2. run propagation with the known graph structure and collected embeddings
        hiddens = self.prop_model(data, batches_seen=self.global_step) # always B x T x N x F
        if self.hparams.hetero_graph:
            hiddens = hiddens.view(len(to_batch_graph_list), -1, hiddens.shape[1], hiddens.shape[2]).permute(0, 2, 1, 3) # B x T x N x F
        # 3. run decoding on all clients
        for client_i, client in enumerate(self.clients):
            client.local_decode_forward(hiddens[:, :, client_i:client_i+1, :])
        # 4. run eval on all clients
        local_val_results = []
        for client_i, client in enumerate(self.clients):
            local_val_results.append(client.local_validation())
        # 6. aggregate local train results
        log = self.aggregate_local_logs([x['log'] for x in local_val_results])
        return {'progress_bar': log, 'log': log}

    def validation_epoch_end(self, outputs):
        return self.training_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        if self.hparams.hetero_graph:
            edge_index_batch, edge_attr_batch = list(zip(*batch))
        # 1. run encoding on all clients and collect embeddings
        encodings = [] # list of dicts of x, x_attr, y, y_attr (B x T x F)
        for client_i, client in enumerate(self.clients):
            encodings.append(client.local_encode_forward('test'))
        stacked_encodings = {}
        for k in encodings[0]:
            stacked_encodings[k] = torch.cat([enc[k] for enc in encodings], dim=2) # B x T x N x F
        if self.hparams.hetero_graph:
            to_batch_graph_list = []
            for bi in range(stacked_encodings['x'].shape[0]):
                to_batch_graph_list.append(Data(
                    x=stacked_encodings['x'][bi].permute(1, 0, 2), y=stacked_encodings['y'][bi].permute(1, 0, 2),
                    x_attr=stacked_encodings['x_attr'][bi].permute(1, 0, 2), y_attr=stacked_encodings['y_attr'][bi].permute(1, 0, 2),
                    edge_attr=edge_attr_batch[bi], edge_index=edge_index_batch[bi]))
            data = Batch.from_data_list(to_batch_graph_list)
        else:
            data = {'x': stacked_encodings['x'], 'y': stacked_encodings['y'], 
                'x_attr': stacked_encodings['x_attr'], 'y_attr': stacked_encodings['y_attr']}
        # # if required, save node encodings
        # if self.hparams.save_node_encodings_test:
        #     # self.test_encodings = encodings.data.cpu().numpy()
        #     self.test_encodings = {
        #         'input': self.data.x.data.cpu().numpy(),
        #         'encodings': encodings.data.cpu().numpy(),
        #     }
        # 2. run propagation with the known graph structure and collected embeddings
        hiddens = self.prop_model(data, batches_seen=self.global_step) # always B x T x N x F
        if self.hparams.hetero_graph:
            hiddens = hiddens.view(len(to_batch_graph_list), -1, hiddens.shape[1], hiddens.shape[2]).permute(0, 2, 1, 3) # B x T x N x F
        # 3. run decoding on all clients
        for client_i, client in enumerate(self.clients):
            client.local_decode_forward(hiddens[:, :, client_i:client_i+1, :])
        # 4. run eval on all clients
        local_test_results = []
        for client_i, client in enumerate(self.clients):
            local_test_results.append(client.local_test())
        # 6. aggregate local train results
        log = self.aggregate_local_logs([x['log'] for x in local_test_results])
        return {'progress_bar': log, 'log': log}

    def test_epoch_end(self, outputs):
        return self.training_epoch_end(outputs)

