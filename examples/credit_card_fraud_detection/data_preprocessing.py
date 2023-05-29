import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_dataset(data_file_path):
    return pd.read_csv(data_file_path)


def split_dataset(data_file_path, num_clients=50, test_rate=0.2):
    df = pd.read_csv(data_file_path)

    proportions = np.random.dirichlet(np.ones(num_clients), size=1)[0]

    split_dfs = []
    start = 0
    for proportion in proportions:
        size = int(proportion * len(df))
        split_dfs.append(df.iloc[start : start + size])
        start += size

    return split_dfs
    """
    train_tests = []
    for split_df in split_dfs:
        train, test = train_test_split(split_df, test_size=test_rate)
        train_tests.append((train, test))

    return train_tests
    """

def get_loader(dataset_df, batch_size=128, shuffle=True, drop_last=False):
    X_df = dataset_df.iloc[:, :-1]
    y_df = dataset_df.iloc[:, -1]

    X_tensor = torch.tensor(X_df.values, dtype=torch.float32)
    y_tensor = torch.tensor(y_df.values, dtype=torch.float32)

    y_tensor = y_tensor.unsqueeze(1)
    dataset_in_dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(
        dataset_in_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )

# def get_loader(dataset_df, batch_size=128, shuffle=True, drop_last=False):
#     X_df = dataset_df.iloc[:, :-1]
#     y_df = dataset_df.iloc[:, -1]

#     X_tensor = torch.tensor(X_df.values, dtype=torch.float32)
#     y_tensor = torch.tensor(y_df.values, dtype=torch.float32)

#     X_tensor = X_tensor.unsqueeze(1)
#     X_tensor = X_tensor.transpose(
#         1, 2
#     )  # Now X_tensor has shape [batch_size, num_features, seq_length]
#     y_tensor = y_tensor.unsqueeze(1)
#     dataset_in_dataset = TensorDataset(X_tensor, y_tensor)
#     return DataLoader(
#         dataset_in_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
#     )


def save_dataset(datasets, dataset_dir="data"):
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    for i, dataset in enumerate(datasets):
        dataset.to_json(f"data/client_{i}.json", orient="records")
