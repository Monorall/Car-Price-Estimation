import torch.nn as nn


class CarPricePredictor(nn.Module):
    def __init__(self, input_size, output_size, num_layers, hidden_size, dropout_rate):
        super(CarPricePredictor, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 1)]
        )
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        for i in range(self.num_layers - 1):
            x = self.hidden_layers[i](x)
            x = self.relu(x)
            x = self.dropout(x)

        x = self.out(x)

        return x



# import config
# import torch.nn as nn
#
#
# class CarPricePredictor(nn.Module):
#     def __init__(self, input_size=config.INPUT_FEATURES, output_size=config.OUT_FEATURES):
#         super(CarPricePredictor, self).__init__()
#
#         self.fc1 = nn.Linear(input_size, 64)
#         self.fc2 = nn.Linear(64, output_size)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#
#         return x
