import unittest

import torch
import torch.nn.functional as F

from .compresser import DGC_SGD
from .models.basic_cnn import CreditFraudNetMLP


class TestDGCSGD(unittest.TestCase):
    def setUp(self):
        self.model = CreditFraudNetMLP(29)
        self.optimizer = DGC_SGD(self.model.parameters(), lr=0.01, momentum=0.9)

    def test_dgc_sgd(self):
        input = torch.randn(2000, 29)
        target = torch.randn(2000, 1)
        output = self.model(input)
        # Compute loss
        loss = F.binary_cross_entropy(output, target)
        # Backward pass
        loss.backward()
        # Get the uncompressed gradient
        grads = [param.grad.clone() for param in self.model.parameters()]
        print(f"Before Optimizer.step - {grads[0]}")
        # Perform optimization step
        self.optimizer.step()
        # Get the gradient
        compressed_grads = self.optimizer.compressed_grads
        print(f"After Optimizer.step - {compressed_grads[0]}")


if __name__ == '__main__':
    unittest.main()
