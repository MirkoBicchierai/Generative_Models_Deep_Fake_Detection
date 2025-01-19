import torch
import torch.nn as nn


class ClassificationLSTM(nn.Module):
    def __init__(self, hidden_size, input_size, num_classes, layers):
        super(ClassificationLSTM, self).__init__()
        self.layers = layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=layers, bidirectional=True, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size*2)
        self.fc_classification = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        h_t = torch.zeros(2 * self.layers, batch_size, self.hidden_size, device=inputs.device)
        c_t = torch.zeros(2 * self.layers, batch_size, self.hidden_size, device=inputs.device)
        output, (h_n, c_n) = self.lstm(inputs, (h_t, c_t))
        output = torch.mean(output, dim=1)
        output = self.norm(output)
        return self.fc_classification(output)


class ClassificationGRU(nn.Module):
    def __init__(self, hidden_size, input_size, num_classes, layers):
        super(ClassificationGRU, self).__init__()
        self.layers = layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers=layers, bidirectional=True, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.fc_classification = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        h_t = torch.zeros(2 * self.layers, batch_size, self.hidden_size, device=inputs.device)
        output, h_n = self.gru(inputs, h_t)
        output = torch.mean(output, dim=1)
        output = self.norm(output)
        return self.fc_classification(output)

class ClassificationMLP(nn.Module):
    def __init__(self,input_size, output_size, hidden_size, num_classes):
        super(ClassificationMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, inputs):
        x = self.mlp(inputs)
        x = self.head(x)
        return x
