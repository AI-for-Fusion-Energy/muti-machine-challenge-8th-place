import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class LinearModel(nn.Module):
    def __init__(self, input_size = 93,
                       hidden_size=128,
                       num_layers=4,
                       window_size=128,
                       kernel_size=3):
        super(LinearModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sig1 = nn.Sigmoid()


    def forward(self, x):
        output = self.fc1(x)
        output = self.sig1(output)

        return output

class LSTMModel(nn.Module):
    def __init__(self, input_size = 93,
                       hidden_size=128,
                       num_layers=4,
                       window_size=128,
                       kernel_size=3):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                            num_layers = num_layers, bias=True, batch_first=True,
                            dropout = 0.3, bidirectional=False)
        self.fc1 = nn.Linear(hidden_size, 1)
        self.sig1 = nn.Sigmoid()


    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.fc1(output)
        output = self.sig1(output)

        return output

class CNNLSTMModel(nn.Module):
    def __init__(self, input_size = 93,
                       hidden_size=128,
                       num_layers=4,
                       window_size=128,
                       kernel_size=3):
        super(CNNLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.upper_layers=[]
        for i in range(num_layers):
            if i == 0:
                input_size = input_size
            else:
                input_size = hidden_size
            self.upper_layers.append(weight_norm(nn.Conv1d(input_size, hidden_size, kernel_size=kernel_size, padding=kernel_size//2)))
            self.upper_layers.append(nn.ReLU6())
        self.upper_net = nn.Sequential(*self.upper_layers)

        self.lstm = nn.LSTM(input_size = window_size, hidden_size = hidden_size,
                            num_layers = 1, bias=True, batch_first=True,
                            dropout = 0.3, bidirectional=False)
        self.fc1 = nn.Linear(hidden_size, 1)
        self.fc2 = nn.Linear(window_size, 1)
        self.sig1 = nn.Sigmoid()
        self.sig2 = nn.Sigmoid()


    def forward(self, x):
        sequential_list = torch.split(x, 1, dim=1)
        upper_output = []
        for data in sequential_list:
            tmp = self.upper_net(data.permute(0, 2, 1))
            upper_output.append(tmp.permute(0, 2, 1))
        down_input = torch.cat(upper_output, dim=1)
        output, _ = self.lstm(down_input)
        output = self.fc1(output)
        output = self.sig1(output)

        return output

