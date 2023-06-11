# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReLUFeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ReLUFeedForwardNN, self).__init__()

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.layer1(x))
        out = self.relu(self.layer2(out))
        out = self.layer3(out)  # No activation on final layer

        return out


class DropoutFeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_p=0.5):
        super(DropoutFeedForwardNN, self).__init__()

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.layer1(x))
        out = self.dropout(out)
        out = self.relu(self.layer2(out))
        return out


class ThreeLayerDropoutNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_p=0.5):
        super(ThreeLayerDropoutNN, self).__init__()

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        out = self.relu(self.layer1(x))
        out = self.dropout(out)
        out = self.relu(self.layer2(out))
        out = self.dropout(out)
        out = self.layer3(out)
        return out


class BatchNormFeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BatchNormFeedForwardNN, self).__init__()

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)

        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)

        self.layer3 = nn.Linear(hidden_size, output_size)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.layer1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.layer3(out)
        return out


class SimpleCNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.fc = nn.Linear(hidden_size, output_size)
        
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(2)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)  # New line added to permute the dimensions
        out = self.relu(self.conv1(x))
        out = self.maxpool(out)
        out = self.relu(self.conv2(out))
        out = self.maxpool(out)
        out = out.view(out.size(0), -1)  # Flatten the tensor
        out = self.fc(out)
        return out


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=16):
        super(SimpleRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(0) # add dim
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # Initialize hidden state
        out, _ = self.rnn(x, h0)
        out = self.fc(out)
        out = out.squeeze(0) # rm dim
        return out


class TransformerClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=5):
        super(TransformerClassifier, self).__init__()
        
        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(d_model=input_size, nhead=1, dim_feedforward=hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Classifier
        self.classifier = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.transformer_encoder(x)
        #print(f'{out.shape=}')
        out = self.classifier(out)  # You might want to feed only the last output of your sequence into the classifier
        return out


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=16):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(0) # add dim
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # Initialize hidden state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # Initialize cell state
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)
        out = out.squeeze(0) # rm dim
        return out



class ResNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ResNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.shortcut = nn.Linear(input_size, hidden_size)  # A shortcut connection that changes the dimension


    def forward(self, x):
        identity = self.shortcut(x)  # Use the shortcut connection to change the dimension of the input
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        out += identity  # Now these should be able to be added together
        out = F.relu(out)
        out = self.fc3(out)
        return out

