import torch.nn as nn


class CarPricePredictor(nn.Module):
    def __init__(self, input_size, output_size, num_layers=2, hidden_size=317, dropout_rate=0.32):
        super(CarPricePredictor, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        self.relu = nn.ReLU()
        self.fc = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size)] * (num_layers-1))
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)

        for i in range(self.num_layers - 1):
            x = self.hidden_layers[i](x)
            x = self.relu(x)
            x = self.dropout(x)

        x = self.out(x)

        return x
