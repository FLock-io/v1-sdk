import torch
from torch import nn
from torch.nn import functional as F


class CreditFraudNetMLP(nn.Module):
    def __init__(self, num_features, num_classes):
        super(CreditFraudNetMLP, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fc3 = nn.Sequential(
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.fc3(x)
        return output



# Has problems , current no need so fix later
class CreditFraudNetCNN(nn.Module):
    
    def __init__(self, num_features, num_classes):
        super(CreditFraudNetCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(num_features, 32, 2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, 2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(64, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.avg_pool1d(x, x.shape[2])  # Global Average Pooling
        x = x.view(x.shape[0], -1)  # Flatten
        x = self.fc1(x)
        output = self.fc2(x)
        return output