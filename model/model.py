import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


import torch.nn as nn

# 모델 구성
# Applying Deep Learning Approaches for Network Traffic Prediction (ICACCI, 2017) 
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, num_classes)
        self.params = self.lstm.state_dict()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        out, _ = self.lstm(x.unsqueeze(2), (h0, c0))
        last_out = out[:, -1, :]
        out = self.fc1(last_out)
        return out
