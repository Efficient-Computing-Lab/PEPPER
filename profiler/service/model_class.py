# models.py
import torch.nn as nn
import torch.nn.functional as F

class MLPRegressor(nn.Module):
    def __init__(self, num_features, hidden_size=128, num_outputs=2):
        super().__init__()
        self.fc1 = nn.Linear(num_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)