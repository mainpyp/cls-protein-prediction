import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 4)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = self.flatten(x)
        return x


class CNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 6, kernel_size=3, padding=1, stride=1)
        self.pool = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(6, 16, kernel_size=3, padding=1, stride=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 256, 256)
        self.fc_fin = nn.Linear(256, 4)

    def forward(self, x):
        x = F.relu( self.conv1(x) )
        x = self.pool(x)
        x = F.relu( self.conv2(x) )
        x = self.pool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc_fin(x)
        return x