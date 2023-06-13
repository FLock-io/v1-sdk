import torch
import io
from loguru import logger
from flock_sdk import FlockSDK
from models.basic_cnn import CreditFraudNetMLP
from data_preprocessing import get_loader
from pandas import DataFrame
import numpy as np
import random


flock = FlockSDK()


class FlockModel:
    def __init__(
        self,
        classes,
        features,
        fabric_instance=None,
        batch_size=256,
        epochs=1,
        lr=0.03,
        client_id=1,
        output_num_classes=1,
    ):
        """
        Hyper parameters
        """
        self.batch_size = batch_size
        self.epochs = epochs
        self.features = features
        self.classes = classes
        self.class_to_idx = {_class: idx for idx, _class in enumerate(self.classes)}
        self.lr = lr
        self.output_num_classes = output_num_classes

        """
            Device setting
        """
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        self.device = torch.device(device)

    def process_dataset(self, dataset: list[dict], transform=None):
        logger.debug("Processing dataset")
        dataset_df = DataFrame.from_records(dataset)
        return get_loader(
            dataset_df, batch_size=batch_size, shuffle=True, drop_last=False
        )

    def get_starting_model(self):
        # torch.manual_seed(0)
        seed = 0
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        return CreditFraudNetMLP(num_features=self.features, num_classes=1)

    """
    train() should:
    1. Take in the model weights as bytes and load them into your model
    2. If parameters passed are None, initialize them to match your untrained model's parameters (i.e. clean slate)
    3. If needed pre-process the dataset which is passed as a list of rows parsed as dicts
    4. Output the model parameters retrained on the dataset AS BYTES
    """

    def train(self, parameters: bytes | None, dataset: list[dict]) -> bytes:
        data_loader = self.process_dataset(dataset)

        model = self.get_starting_model()
        if parameters is not None:
            model.load_state_dict(torch.load(io.BytesIO(parameters)))
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.lr,
        )
        criterion = torch.nn.BCELoss()
        model.to(self.device)

        # pro_bar = tqdm(range(self.epochs))
        for epoch in range(self.epochs):
            logger.debug(f"Epoch {epoch}")
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            for batch_idx, (inputs, targets) in enumerate(data_loader):
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
                if batch_idx < 2:
                    logger.debug(
                        f"Batch {batch_idx}, Acc: {round(100.0 * train_correct / train_total, 2)}, Loss: {round(train_loss / train_total, 4)}"
                    )

            logger.info(
                f"Training Epoch: {epoch}, Acc: {round(100.0 * train_correct / train_total, 2)}, Loss: {round(train_loss / train_total, 4)}"
            )

        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        return buffer.getvalue()

    """
    evaluate() should:
    1. Take in the model weights as bytes and load them into your model
    3. If parameters passed are None, initialize them to match your untrained model's parameters (i.e. clean slate)
    4. If needed pre-process the dataset which is passed as a list of rows parsed as dicts
    5. Output the accuracy of the model parameters on the dataset as a float
    """

    def evaluate(self, parameters: bytes | None, dataset: list[dict]) -> float:
        data_loader = self.process_dataset(dataset)
        criterion = torch.nn.BCELoss()

        model = self.get_starting_model()
        if parameters is not None:
            model.load_state_dict(torch.load(io.BytesIO(parameters)))
        model.to(self.device)
        model.eval()

        test_correct = 0
        test_loss = 0.0
        test_total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item() * inputs.size(0)
                predicted = torch.round(outputs).squeeze()
                test_total += targets.size(0)
                test_correct += (predicted == targets.squeeze()).sum().item()

        accuracy = test_correct / test_total
        logger.info(
            f"Model test, Acc: {accuracy}, Loss: {round(test_loss / test_total, 4)}"
        )

        return accuracy

    """
    aggregate() should take in a list of model weights (bytes),
    aggregate them using avg and output the aggregated parameters as bytes.
    """

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
    """
    Hyper parameters
    """
    batch_size = 128
    epochs = 1
    # lr = 0.00000001 Too low
    # lr = 0.000001  Too high
    # lr = 0.0000001 Too low
    # lr = 0.00000035 Better
    lr = 0.000001
    classes = [
        "0",
        "1",
    ]

    flock_model = FlockModel(
        classes,
        30,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
    )
    flock.register_train(flock_model.train)
    flock.register_evaluate(flock_model.evaluate)
    flock.register_aggregate(flock_model.aggregate)
    flock.run()