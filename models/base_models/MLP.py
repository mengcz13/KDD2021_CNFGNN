from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout, final_activation=False, *args, **kwargs):
        super().__init__()
        layers = []
        last_size = input_size
        for t in range(num_layers - 1):
            if t == 0:
                layers.append(nn.Linear(input_size, hidden_size))
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Dropout(p=dropout))
            layers.append(nn.ReLU())
            last_size = hidden_size
        layers.append(nn.Linear(last_size, output_size))
        if final_activation:
            layers.append(nn.Dropout(p=dropout))
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, data):
        return self.layers(data.x)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden_size', type=int, default=128)
        parser.add_argument('--num_layers', type=int, default=3)
        parser.add_argument('--dropout', type=float, default=0)
        return parser