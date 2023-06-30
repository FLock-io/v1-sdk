import copy
import json

import torch
import torch.utils.data
import torch.utils.data.distributed
from loguru import logger
from pandas import DataFrame

from data_preprocessing import get_loader
from flock_sdk import FlockSDK
from models.basic_cnn import CreditFraudNetMLP

flock = FlockSDK()
import random
import numpy as np


def dgc(grads, sparsity: float = 0.9):
    """
    This function implements the Deep Gradient Compression (DGC) for a given set of gradients.
    Args:
    grads: list of gradients to compress
    sparsity: the desired sparsity level, a float between 0 and 1.
    """
    # Flatten the gradients into a single 1-D tensor
    flat_grads = torch.cat([grad.view(-1) for grad in grads])

    # Compute the threshold
    abs_grads = flat_grads.abs()
    k = int(sparsity * flat_grads.numel())
    threshold = abs_grads.topk(k, largest=False).values.max()

    # Create a mask for the elements to keep
    mask = abs_grads.gt(threshold).float()

    # Apply the mask to the original gradients
    compressed_grads = []
    start = 0
    for grad in grads:
        end = start + grad.numel()
        compressed_grad = grad * mask[start:end].view_as(grad)
        compressed_grads.append(compressed_grad)
        start = end
    sparse_tensors = [compressed_grad.to_sparse() for compressed_grad in compressed_grads]

    return sparse_tensors


def compare_models(model1, model2):
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if name1 == name2:
            print(f'Parameter: {name1}')
            print(f'Difference: {torch.sum(param1.data - param2.data)}')


def get_model():
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    return CreditFraudNetMLP(num_features=features, num_classes=1)


batch_size = 128
epochs = 100
lr = 0.0003
device = torch.device('cpu')
features = 29


def process_dataset(dataset: list[dict], transform=None):
    logger.debug('Processing dataset')
    dataset_df = DataFrame.from_records(dataset)
    return get_loader(
        dataset_df, batch_size=batch_size, shuffle=True, drop_last=False
    )


with open('test_dataset_new.json', 'r') as f:
    dataset = json.loads(f.read())

data_loader = process_dataset(dataset)

train_model = get_model()
eval_model_1 = get_model()
eval_model_2 = get_model()

# compare_models(train_model, eval_model_1)


train_model.train()
train_model.to(device)
eval_model_1.to(device)
eval_model_2.to(device)

optimizer = torch.optim.SGD(train_model.parameters(), lr=lr)
criterion = torch.nn.BCELoss()

uncompressed_grads_accumulate = None
compressed_grads_accumulate = None

for epoch in range(epochs):
    logger.debug(f'Epoch {epoch}')
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    for batch_idx, (inputs, targets) in enumerate(data_loader):

        # remove Time parameter
        inputs, targets = inputs.to(device)[:, 1:], targets.to(device)
        outputs = train_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        # self.fabric.backward(loss)
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        predicted = torch.round(outputs).squeeze()
        train_total += targets.size(0)
        train_correct += (predicted == targets.squeeze()).sum().item()

        uncompressed_grads = [p.grad.data for p in train_model.parameters()]
        if uncompressed_grads_accumulate is None:
            uncompressed_grads_accumulate = copy.deepcopy(uncompressed_grads)
        else:
            for grads_accumulate, uncompressed_grad in zip(uncompressed_grads_accumulate, uncompressed_grads):
                grads_accumulate += uncompressed_grad

        compressed_grads = dgc(uncompressed_grads)
        if compressed_grads_accumulate is None:
            compressed_grads_accumulate = copy.deepcopy(compressed_grads)
        else:
            for grads_accumulate, compressed_grad in zip(compressed_grads_accumulate, compressed_grads):
                grads_accumulate += compressed_grad

    logger.info(
        f'Training Epoch: {epoch}, Acc: {round(100.0 * train_correct / train_total, 2)}, Loss: {round(train_loss / train_total, 4)}'
    )

for p, compressed_grad in zip(eval_model_1.parameters(), uncompressed_grads_accumulate):
    # p.data.add_(-self.lr * compressed_grad)
    p.data -= lr * compressed_grad

for p, compressed_grad in zip(eval_model_2.parameters(), compressed_grads_accumulate):
    # p.data.add_(-self.lr * compressed_grad)
    p.data -= lr * compressed_grad

# print('==============================================')
compare_models(train_model, eval_model_1)
print('==============================================')
