import torch
import torch.nn as nn
import random
import os

class RandomLinearQNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, hidden_size):
        super().__init__()
        print("input_size: ", input_size)
        print("output_size: ", output_size)
        print("hidden_layers: ", hidden_layers)
        print("hidden_size: ", hidden_size)

        layers = []
        for i in range(hidden_layers):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_size))
            else:
                layers.append(nn.Linear(prev_hidden_size, hidden_size))

            layers.append(nn.ReLU())
            prev_hidden_size = hidden_size

        layers.append(nn.Linear(prev_hidden_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, t):
        return self.net(t)

