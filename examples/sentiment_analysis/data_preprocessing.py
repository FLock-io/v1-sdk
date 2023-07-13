
import torch
import re
import csv
from torch.utils.data import Dataset, DataLoader

class IndexesDataset(Dataset):
    def __init__(self, dataset, vocab=None, max_seq_len=64, device="cuda", max_samples_count=20000,max_vocab_size=30000):
        self.samples = []
        self.labels = []
        self.max_seq_len = max_seq_len
        self.device = device

        for row in dataset:
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
            self.vocab = self._make_vocab(max_vocab_size=max_vocab_size)

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
        for sample in self.samples:
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

def get_loader(dataset_df, batch_size):

    return DataLoader(dataset_df, batch_size=batch_size, num_workers=1, collate_fn=dataset_df.collate)