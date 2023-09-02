import json
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
import io
import torch
from pandas import DataFrame
from flock_sdk import FlockSDK, FlockModel


class CreditFraudNetMLP(nn.Module):
    def __init__(self, num_features, num_classes):
        super(CreditFraudNetMLP, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(num_features, 64), nn.ReLU(), nn.Dropout(0.2)
        )

        self.fc2 = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.5))

        self.fc3 = nn.Sequential(nn.Linear(128, num_classes), nn.Sigmoid())

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.fc3(x)
        return output


class ExampleTorchModel(FlockModel):
    def __init__(
        self,
        features,
        epochs=1,
        lr=0.03,
    ):
        """
        Hyper parameters
        """
        self.epochs = epochs
        self.features = features
        self.lr = lr

        """
            Device setting
        """
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        self.device = torch.device(device)

    def init_dataset(self, dataset_path: str) -> None:
        self.dataset_path = dataset_path
        with open(dataset_path, "r") as f:
            dataset = json.load(f)
        dataset_df = DataFrame.from_records(dataset)

        train_ratio = 0.5
        batch_size = 128
        train_len = int(len(dataset) * train_ratio)
        test_len = len(dataset) - train_len

        # Split the dataset
        train_dataset, test_dataset = random_split(dataset_df, [train_len, test_len])

        X_df = dataset_df.iloc[:, :-1]
        y_df = dataset_df.iloc[:, -1]

        X_tensor = torch.tensor(X_df.values, dtype=torch.float32)
        y_tensor = torch.tensor(y_df.values, dtype=torch.float32)

        y_tensor = y_tensor.unsqueeze(1)
        dataset_in_dataset = TensorDataset(X_tensor, y_tensor)
        self.train_data_loader = DataLoader(
            dataset_in_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        self.test_data_loader = DataLoader(
            dataset_in_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        # Create data loaders
        """
        self.train_data_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.test_data_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True
        )
        """

    def get_model(self):
        return CreditFraudNetMLP(num_features=self.features, num_classes=1)

    def train(self, parameters) -> bytes:
        model = self.get_model()
        if parameters != None:
            model.load_state_dict(torch.load(io.BytesIO(parameters)))

        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.lr,
        )
        criterion = torch.nn.BCELoss()
        model.to(self.device)

        for epoch in range(self.epochs):
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            for inputs, targets in self.train_data_loader:
                optimizer.zero_grad()

                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)

                loss = criterion(outputs, targets)
                loss.backward()

                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                predicted = torch.round(outputs).squeeze()
                train_total += targets.size(0)
                train_correct += (predicted == targets.squeeze()).sum().item()

            print(
                f"Training Epoch: {epoch}, Acc: {round(100.0 * train_correct / train_total, 2)}, Loss: {round(train_loss / train_total, 4)}"
            )

        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        return buffer.getvalue()

    def evaluate(self, parameters: bytes) -> float:
        criterion = torch.nn.BCELoss()

        model = self.get_model()
        if parameters != None:
            model.load_state_dict(torch.load(io.BytesIO(parameters)))
        model.to(self.device)
        model.eval()

        test_correct = 0
        test_loss = 0.0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in self.test_data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item() * inputs.size(0)
                predicted = torch.round(outputs).squeeze()
                test_total += targets.size(0)
                test_correct += (predicted == targets.squeeze()).sum().item()

        accuracy = test_correct / test_total
        return accuracy

    def aggregate(self, parameters_list: list[bytes]) -> bytes:
        parameters_list = [
            torch.load(io.BytesIO(parameters)) for parameters in parameters_list
        ]
        averaged_params_template = parameters_list[0]
        for k in averaged_params_template.keys():
            temp_w = []
            for local_w in parameters_list:
                temp_w.append(local_w[k])
            averaged_params_template[k] = sum(temp_w) / len(temp_w)

        # Create a buffer
        buffer = io.BytesIO()

        # Save state dict to the buffer
        torch.save(averaged_params_template, buffer)

        # Get the byte representation
        aggregated_parameters = buffer.getvalue()

        return aggregated_parameters


if __name__ == "__main__":
    epochs = 1
    lr = 0.000001
    features = 30
    model = ExampleTorchModel(features, epochs=epochs, lr=lr)
    sdk = FlockSDK(model)
    sdk.run()
