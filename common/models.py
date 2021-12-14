import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """ Define a model with two convolution layers each followed by a Max Pooling layer
    Then the embeddings are Flattened and forwarded into two FCls.
    The last layer is a FCL with 10 neurons.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 512)  # in channels -> 16 * 5 ^ 2 (kernel size 5)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 4)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        #x = F.softmax(x, dim=4)
        return x


class CNN(nn.Module):
    """ Define a model with two convolution layers each followed by a Max Pooling layer
    Then the embeddings are Flattened and forwarded into two FCls.
    The last layer is a FCL with 10 neurons.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 6, kernel_size=3, padding=1, stride=1)
        self.pool = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(6, 16, kernel_size=3, padding=1, stride=1)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(16 * 256, 256)

        self.fc_fin = nn.Linear(256, 4)
    """
    view ->   torch.Size([4, 1, 1024])
    view conv1 +  batch ->   torch.Size([4, 512, 1024])
    view pool 1 ->   torch.Size([4, 512, 512])
    view conv2 +  batch ->   torch.Size([4, 256, 512])
    view pool 2 ->   torch.Size([4, 256, 256])
    """
    def forward(self, x):
        x = F.relu( self.conv1(x) )
        x = self.pool(x)

        x = F.relu( self.conv2(x) )
        x = self.pool(x)

        x = self.flatten(x)

        x = F.relu(self.fc1(x))

        x = self.fc_fin(x)
        return x