from torch import nn

class CreditFraudNetMLP(nn.Module):
    def __init__(self, num_features, num_classes):
        super(CreditFraudNetMLP, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(num_features, 64), nn.ReLU(), nn.Dropout(0.2))
        self.fc2 = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.5))
        self.fc3 = nn.Sequential(nn.Linear(128, num_classes), nn.Sigmoid())

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    
    