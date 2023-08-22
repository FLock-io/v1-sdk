import re
import csv
from tqdm import tqdm
import copy
import torch
from torch import nn
from models.basic_cnn import CNNClassifier,LinearClassifier
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from sklearn.metrics import accuracy_score

def dgc(grads, sparsity: float = 0.95):
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

class IndexesDataset(Dataset):
    def __init__(self, path, vocab=None, max_seq_len=64, device="cuda", max_samples_count=20000):
        self.samples = []
        self.labels = []
        self.max_seq_len = max_seq_len
        self.device = device

        with open(path) as train_file:
            reader = csv.reader(train_file)

            next(reader)

            for row in tqdm(reader, desc="Load data"):
                label = row[0]
                sample = row[1]

                sample = re.sub(r"([.,!?'])", r" \1", sample)
                sample = re.sub(r"[^a-zA-Z0-9.,!?']", " ", sample)

                self.labels.append(int(label) - 1)
                self.samples.append(sample)

                if len(self.samples) > max_samples_count:
                    break

        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = self._make_vocab()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        sample = [self.get_index(word) for word in sample.split()]

        sample = sample[:self.max_seq_len]
        pad_len = self.max_seq_len - len(sample)
        sample += [self.get_index("[PAD]")] * pad_len

        label = self.labels[index]

        return sample, label

    def _make_vocab(self, max_vocab_size=30000):
        vocab = {"[PAD]": 1000000000000001, "[UNK]": 100000000000000}
        for sample in tqdm(self.samples, desc="Make vocab"):
            for word in sample.split():
                if word not in vocab.keys():
                    vocab[word] = 1
                else:
                    vocab[word] += 1

        vocab = dict(sorted(vocab.items(), key=lambda item: item[1], reverse=True))

        vocab = list(vocab.keys())[:max_vocab_size]
        return vocab

    def get_vocab(self):
        return self.vocab

    def get_index(self, word):
        if word in self.vocab:
            index = self.vocab.index(word)
        else:
            index = self.vocab.index("[UNK]")

        return index

    def collate(self, batch):
        input_ids, targets = list(zip(*batch))
        return torch.tensor(input_ids), torch.tensor(targets, dtype=torch.float32)

def compare_models(model1, model2):
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if name1 == name2:
            print(f'Parameter: {name1}')
            print(f'Difference: {torch.sum(param1.data - param2.data)}')

if __name__ == '__main__':

    device = "cuda"
    max_seq_len = 64
    epoches = 3
    lr =0.001
    train_path = "train.csv"
    test_path = "test.csv"

    train_indexes_dataset = IndexesDataset(train_path, max_samples_count=10000, device=device)
    vocab = train_indexes_dataset.get_vocab()
    vocab_size = len(vocab)
    print(f"Vocab_size: {vocab_size}")
    test_indexes_dataset = IndexesDataset(test_path, vocab=vocab, max_samples_count=2000, device=device)

    train_indexes_dataloader = DataLoader(train_indexes_dataset, batch_size=64, num_workers=1, collate_fn=train_indexes_dataset.collate)
    test_indexes_dataloader = DataLoader(test_indexes_dataset, batch_size=64, num_workers=1, collate_fn=train_indexes_dataset.collate)

    cnn_classifier = LinearClassifier(vocab_size, 100)
    model1 =copy.deepcopy(cnn_classifier)
    model2 = copy.deepcopy(cnn_classifier)

    cnn_classifier = cnn_classifier.to(device)
    model1 = model1.to(device)
    model2 = model2.to(device)
    print('==============================================')
    compare_models(cnn_classifier, model1)

    optimizer = optim.SGD(cnn_classifier.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    uncompressed_grads_accumulate = None
    compressed_grads_accumulate = None

    for i in range(epoches):
        cnn_classifier.train()
        train_loss = 0
        train_count = 0
        for batch in tqdm(train_indexes_dataloader, desc="Training"):
            optimizer.zero_grad()
            input_ids, targets = batch
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            logits = cnn_classifier(input_ids)
            loss = loss_fn(logits.squeeze(1), targets)
            loss.backward()
            uncompressed_grads = [p.grad.clone() for p in cnn_classifier.parameters()]
            optimizer.step()

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

            train_loss += loss.item()
            train_count += 1
        print(f"Train loss: {train_loss / train_count:.3f}")
    # for p, compressed_grad in zip(model1.parameters(), uncompressed_grads_accumulate):
    #     # p.data.add_(-self.lr * compressed_grad)
    #     p.data -= lr * compressed_grad
    #
    # for p, compressed_grad in zip(model2.parameters(), compressed_grads_accumulate):
    #     # p.data.add_(-self.lr * compressed_grad)
    #     p.data -= lr * compressed_grad
    #
    # print('==============================================')
    # compare_models(cnn_classifier, model1)
    # print('==============================================')
    # compare_models(cnn_classifier, model2)

        # cnn_classifier.eval()
        # all_preds = []
        # all_targets = []
        # eval_loss = 0
        # eval_count = 0
        # for batch in tqdm(test_indexes_dataloader, desc="Evaluation"):
        #     input_ids, targets = batch
        #     input_ids = input_ids.to(device)
        #     targets = targets.to(device)
        #     logits = cnn_classifier(input_ids)
        #     loss = loss_fn(logits.squeeze(1), targets)
        #     probs = torch.sigmoid(logits.squeeze(1)).tolist()
        #     preds = [0 if p < 0.5 else 1 for p in probs]
        #     all_preds += preds
        #     all_targets += targets.cpu().tolist()
        #
        #     eval_loss += loss.item()
        #     eval_count += 1
        #
        # acc = accuracy_score(all_targets, all_preds)
        # print(f"Train loss: {train_loss/train_count:.3f} Eval loss: {eval_loss/eval_count:.3f} Eval Acc: {acc:.2f}")

