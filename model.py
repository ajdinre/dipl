# model.py

import torch.nn as nn

class Model1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model1, self).__init__()

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.layer1(x))
        out = self.relu(self.layer2(out))
        out = self.layer3(out)  # No activation on final layer

        return out
    

class DOModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DOModel, self).__init__()

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.dropout(self.relu(self.layer1(x)))
        out = self.dropout(self.relu(self.layer2(out)))
        out = self.layer3(out)  # No activation on final layer

        return out


class BNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BNModel, self).__init__()

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn(self.layer1(x)))
        out = self.relu(self.bn(self.layer2(out)))
        out = self.layer3(out)  # No activation on final layer

        return out

