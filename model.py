import torch
import torch.nn as nn
# LSTM Model Definition
class ComplexDiabetesLSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size=64, output_size=1, num_layers=3, dropout_rate=0.5):
        super(ComplexDiabetesLSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers

        # LSTM with multiple layers and dropout
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, batch_first=True, dropout=dropout_rate)

        # Additional Linear layers
        self.linear1 = nn.Linear(hidden_layer_size, 32)
        self.linear2 = nn.Linear(32, output_size)

        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        lstm_out = lstm_out[:, -1, :]  # Take the output of the last time step

        out = self.dropout(self.relu(self.linear1(lstm_out)))
        predictions = torch.sigmoid(self.linear2(out))

        return predictions
