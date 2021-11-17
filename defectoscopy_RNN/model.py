import torch
import torch.nn as nn


class CnnLstm(nn.Module):
    def __init__(self, num_layers, bidirectional, dropout,
                 input_size, hidden_layer_size, output_size):
        super(CnnLstm, self).__init__()

        self.hidden_layer_size = hidden_layer_size
        self.conv2d = nn.Conv2d(1, 1, (3, 3), stride=2)
        self.bn2d = nn.BatchNorm2d(1)
        self.bn1d = nn.BatchNorm1d(195)
        self.avrg2d = nn.AvgPool2d((3, 3), stride=2)
        self.linear_cnn = nn.Linear(195, input_size)
        self.bidirectional = bidirectional
        self.num_layers = num_layers

        # CNN
        self.seq_conv = nn.ModuleList()

        for idx in range(2):
            self.seq_conv.append(self.conv2d)
            self.seq_conv.append(self.bn2d)
            self.seq_conv.append(nn.ReLU())
            self.seq_conv.append(self.avrg2d)

        self.seq_conv.append(nn.Flatten())
        self.seq_conv.append(self.bn1d)
        self.seq_conv.append(self.linear_cnn)

        # LSTM
        self.lstm_list = nn.LSTM(input_size=input_size, hidden_size=hidden_layer_size,
                                 num_layers=self.num_layers, bidirectional=bidirectional,
                                 dropout=dropout)
        if self.bidirectional:
            self.linear_lstm = nn.Linear(hidden_layer_size * 2, output_size)
        else:
            self.linear_lstm = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)

        # CNN
        for layer in self.seq_conv:
            x = layer(x)

        # LSTM
        if self.bidirectional:
            hidden_cell = (torch.zeros(self.num_layers * 2, 1, self.hidden_layer_size),
                                torch.zeros(self.num_layers * 2, 1, self.hidden_layer_size))
        else:
            hidden_cell = (torch.zeros(self.num_layers, 1, self.hidden_layer_size),
                                torch.zeros(self.num_layers, 1, self.hidden_layer_size))

        lstm_out, hidden_cell = self.lstm_list(x.view(len(x), 1, -1), hidden_cell)
        predictions = self.linear_lstm(lstm_out.view(len(x), -1))
        return predictions
