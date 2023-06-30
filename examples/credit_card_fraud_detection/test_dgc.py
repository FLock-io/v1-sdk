import unittest

import torch
import torch.nn.functional as F

from .compresser import DGC_SGD, dgc_func
from .models.basic_cnn import CreditFraudNetMLP


class TestDGCSGD(unittest.TestCase):
    def setUp(self):
        self.model = CreditFraudNetMLP(29)
        self.optimizer = DGC_SGD(self.model.parameters(), lr=0.01, momentum=0.9)

    def test_dgc_sgd(self):
        inputs = torch.randn(2000, 29)
        target = torch.randn(2000, 1)
        output = self.model(inputs)
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

    def test_dgc_sgd_save_all_grad(self):
        # Save the 200 batch gradients, each batch has 200 samples. NO DGC gradient update.
        features = torch.randn(200, 200, 29)
        targets = torch.randn(200, 200, 1)
        datasets = zip(features, targets)
        # save all gradients
        all_grads = []
        for feature, target in datasets:
            outs = self.model(feature)
            loss_val = F.binary_cross_entropy(outs, target)
            loss_val.backward()
            uncompressed_grad = [param.grad.clone() for param in self.model.parameters()]
            compressed_grad = dgc_func(uncompressed_grad)
            all_grads.append(uncompressed_grad)
        torch.save(all_grads, "all_uncompressed_grads.pt")


if __name__ == '__main__':
    unittest.main()
