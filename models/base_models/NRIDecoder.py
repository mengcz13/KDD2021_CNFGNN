from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.data import Batch


class NRIDecoderUnit(MessagePassing):
    def __init__(self, input_size, hidden_size, output_size, *args, **kwargs):
        super().__init__(aggr='add', node_dim=0)
        self.message_net = nn.Sequential(
            nn.Linear(2 * input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.update_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x, edge_index):
        delta_x = self.propagate(edge_index, x=x)
        return x + delta_x

    def message(self, x_i, x_j):
        return self.message_net(torch.cat((x_i, x_j), dim=-1))

    def update(self, x_out):
        return self.update_net(x_out)


class NRIDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout,
        cl_decay_steps, use_curriculum_learning, decoder_unit_type,
        *args, **kwargs):
        super().__init__()
        self.cl_decay_steps = cl_decay_steps
        self.use_curriculum_learning = use_curriculum_learning
        self.decoder_unit_type = decoder_unit_type
        if decoder_unit_type == 'gru':
            self.decoder = nn.GRU(
                input_size, hidden_size, num_layers=2, dropout=dropout
            )
            self.out_net = nn.Linear(hidden_size, output_size)
        elif decoder_unit_type == 'mpnn':
            self.decoder = NRIDecoderUnit(input_size, hidden_size, output_size)

    def _compute_sampling_threshold(self, batches_seen):
        if self.cl_decay_steps == 0:
            return 0
        else:
            return self.cl_decay_steps / (
                    self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def forward(self, data, batches_seen):
        # B x T x N x F
        x, x_attr, y, y_attr = data['x'], data['x_attr'], data['y'], data['y_attr']
        if type(data) is Batch: # N x T x F
            x = x.permute(1, 0, 2).unsqueeze(0)
            x_attr = x_attr.permute(1, 0, 2).unsqueeze(0)
            y = y.permute(1, 0, 2).unsqueeze(0)
            y_attr = y_attr.permute(1, 0, 2).unsqueeze(0)
        batch_num, node_num = x.shape[0], x.shape[2]
        x_input = torch.cat((x, x_attr), dim=-1).permute(1, 0, 2, 3).flatten(1, 2) # T x (B x N) x F
        if self.training and (not self.use_curriculum_learning):
            y_input = torch.cat((y, y_attr), dim=-1).permute(1, 0, 2, 3).flatten(1, 2)
            y_input = torch.cat((x_input[-1:], y_input[:-1]), dim=0)
            if self.decoder_unit_type == 'gru':
                out_hidden, _ = self.decoder(y_input)
                out = self.out_net(out_hidden) + y_input
            elif self.decoder_unit_type == 'mpnn':
                y_input = y_input.permute(1, 0, 2)
                out = self.decoder(y_input, data['edge_index'])
                out = out.permute(1, 0, 2)
            out = out.view(out.shape[0], batch_num, node_num, out.shape[-1]).permute(1, 0, 2, 3)
        else:
            last_input = x_input[-1:]
            last_hidden = None
            step_num = y_attr.shape[1]
            out_steps = []
            y_input = y.permute(1, 0, 2, 3).flatten(1, 2)
            y_attr_input = y_attr.permute(1, 0, 2, 3).flatten(1, 2)
            for t in range(step_num):
                if self.decoder_unit_type == 'gru':
                    out_hidden, last_hidden = self.decoder(last_input, last_hidden)
                    out = self.out_net(out_hidden) # T x (B x N) x F
                elif self.decoder_unit_type == 'mpnn':
                    last_input = last_input.permute(1, 0, 2)
                    out = self.decoder(last_input, data['edge_index'])
                    out = out.permute(1, 0, 2)
                    last_input = last_input.permute(1, 0, 2)
                out_steps.append(out)
                last_input_from_output = torch.cat((out, y_attr_input[t:t+1]), dim=-1)
                last_input_from_gt = torch.cat((y_input[t:t+1], y_attr_input[t:t+1]), dim=-1)
                if self.training:
                    p_gt = self._compute_sampling_threshold(batches_seen)
                    p = torch.rand(1).item()
                    if p <= p_gt:
                        last_input = last_input_from_gt
                    else:
                        last_input = last_input_from_output
                else:
                    last_input = last_input_from_output
            out = torch.cat(out_steps, dim=0)
            out = out.view(out.shape[0], batch_num, node_num, out.shape[-1]).permute(1, 0, 2, 3)
        if type(data) is Batch:
            out = out.squeeze(0).permute(1, 0, 2) # N x T x F
        return out

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden_size', type=int, default=128)
        parser.add_argument('--dropout', type=float, default=0)
        parser.add_argument('--cl_decay_steps', type=int, default=1000)
        parser.add_argument('--use_curriculum_learning', action='store_true')
        parser.add_argument('--decoder_unit_type', default='mpnn')
        return parser