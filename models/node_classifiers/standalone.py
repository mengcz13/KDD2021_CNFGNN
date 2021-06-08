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


class NodeClassifier(LightningModule):
    def __init__(self, hparams, *args, **kwargs):
        super().__init__()
        self.hparams = hparams
        self.base_model = None
        self.base_model_class = getattr(base_models, self.hparams.base_model_name)
        self.setup(None)

    def forward(self, x):
        return self.base_model(x)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--weight_decay', type=float, default=0.0)
        parser.add_argument('--batch_size', type=int, default=128)
        return parser

    def prepare_data(self):
        data = load_dataset(dataset_name=self.hparams.dataset, split_edges=False)

    def setup(self, step):
        if self.base_model is not None:
            return
        data = load_dataset(dataset_name=self.hparams.dataset, split_edges=False)
        self.data = data
        self.base_model = self.base_model_class(
            input_size=data.x.shape[-1],
            output_size=data.y.unique().shape[0],
            **self.hparams
        )

    def train_dataloader(self):
        return DataLoader([self.data], batch_size=self.hparams.batch_size,
            num_workers=1)

    def val_dataloader(self):
        return DataLoader([self.data], batch_size=self.hparams.batch_size,
            num_workers=1)

    def test_dataloader(self):
        return DataLoader([self.data], batch_size=self.hparams.batch_size,
            num_workers=1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        m_out = self(batch)
        y_prob = nn.Softmax(dim=1)(m_out)
        y_pred = y_prob.argmax(dim=1)
        mask = batch.train_mask
        loss = nn.CrossEntropyLoss()(m_out[mask], batch.y[mask])
        accu = (y_pred[mask] == batch.y[mask]).float().mean()

        log = {'train/loss': loss, 'train/accu': accu, 'num': y_pred[mask].shape[0]}
        return {'loss': loss, 'progress_bar': log, 'log': log}

    def training_epoch_end(self, outputs):
        # average all statistics (weighted by sample counts)
        log = {}
        for output in outputs:
            for k in output['log']:
                if k not in log:
                    log[k] = 0
                if k == 'num':
                    log[k] += output['log'][k]
                else:
                    log[k] += (output['log'][k] * output['log']['num'])
        for k in log:
            if k != 'num':
                log[k] = log[k] / log['num']
        log.pop('num')
        return {'log': log, 'progress_bar': log}

    def validation_step(self, batch, batch_idx):
        m_out = self(batch)
        y_prob = nn.Softmax(dim=1)(m_out)
        y_pred = y_prob.argmax(dim=1)
        mask = batch.val_mask
        loss = nn.CrossEntropyLoss()(m_out[mask], batch.y[mask])
        accu = (y_pred[mask] == batch.y[mask]).float().mean()

        log = {'val/loss': loss, 'val/accu': accu, 'num': y_pred[mask].shape[0]}
        return {'loss': loss, 'progress_bar': log, 'log': log}

    def validation_epoch_end(self, outputs):
        return self.training_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        m_out = self(batch)
        y_prob = nn.Softmax(dim=1)(m_out)
        y_pred = y_prob.argmax(dim=1)
        mask = batch.test_mask
        loss = nn.CrossEntropyLoss()(m_out[mask], batch.y[mask])
        accu = (y_pred[mask] == batch.y[mask]).float().mean()

        log = {'test/loss': loss, 'test/accu': accu, 'num': y_pred[mask].shape[0]}
        return {'loss': loss, 'progress_bar': log, 'log': log}

    def test_epoch_end(self, outputs):
        return self.training_epoch_end(outputs)