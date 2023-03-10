import config
import torch.nn as nn


class CarPricePredictor(nn.Module):
    def __init__(self, input_size=config.INPUT_FEATURES, output_size=config.OUT_FEATURES):
        super(CarPricePredictor, self).__init__()

        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
