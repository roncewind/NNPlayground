# Neural Network Model Definition supporting multiple hidden layers
import torch.nn as nn
from config import HIDDEN_SIZE, NUM_CLASSES, NUM_INPUTS, NUM_LAYERS


class NeuralNet(nn.Module):
    def __init__(self, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS):
        super().__init__()
        layers = []

        # First layer: input → hidden
        layers.append(nn.Linear(NUM_INPUTS, hidden_size))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_size, NUM_CLASSES))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# Original NeuralNet class for reference
# import torch.nn as nn
# from config import NUM_CLASSES, NUM_INPUTS


# class NeuralNet(nn.Module):
#     def __init__(self, hidden_size=64):
#         super().__init__()
#         self.fc1 = nn.Linear(NUM_INPUTS, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.fc3 = nn.Linear(hidden_size, NUM_CLASSES)

#     def forward(self, x):
#         x = nn.functional.relu(self.fc1(x))
#         x = nn.functional.relu(self.fc2(x))
#         return self.fc3(x)
