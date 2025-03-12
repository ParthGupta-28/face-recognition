from torch import nn
import torch
import torch.nn.functional as F

class FaceRecogModel(nn.Module):
    def __init__(self):
        super(FaceRecogModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.linear_dims = None
        x = torch.randn(1, 3, 128, 128)
        self.convs(x)
        self.fc1 = nn.Linear(self.linear_dims, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.flatten = nn.Flatten()

    def convs(self ,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        if self.linear_dims is None:
            self.linear_dims = x.shape[1]*x.shape[2]*x.shape[3]
            print(self.linear_dims)
        return x

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x