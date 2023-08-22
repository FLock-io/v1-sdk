from torch import nn
import torch.nn.functional as F

class LinearClassifier(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.linear = nn.Linear(emb_size, 1)

    def forward(self, x):
        embs = self.embedding(x)
        embs = embs.mean(1)
        logits = self.linear(embs)
        return logits

class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.conv1 = nn.Conv1d(emb_size, 100, 5, padding=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(100, 100, 5, padding=2)
        self.relu2 = nn.ReLU()

        self.linear = nn.Linear(100, 1)

    def forward(self, x):
        embs = self.embedding(x)
        h = self.relu1(self.conv1(embs.transpose(1, 2)))
        h = self.relu2(self.conv2(h))

        h_size = h.size(dim=2)
        h = F.avg_pool1d(h, h_size).squeeze(dim=2)

        logits = self.linear(h)

        return logits
