import torch
import io
from loguru import logger
from flock_sdk import FlockSDK
import copy

import torch.utils.data
import torch.utils.data.distributed
from data_preprocessing import load_dataset, get_loader
from models.basic_cnn import CreditFraudNetMLP
from tqdm import tqdm
from pandas import DataFrame

flock = FlockSDK()


class FlockModel:
    def __init__(
        self,
        classes,
        features,
        image_size=84,
        batch_size=256,
        epochs=1,
        lr=0.03,
        compression_method="dgc",
        compress_ratio=0.3,
    ):
        """
        Hyper parameters
        """
        self.features = features
        self.image_size = image_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.classes = classes
        self.class_to_idx = {_class: idx for idx, _class in enumerate(self.classes)}
        self.lr = lr

        """
            Data prepare
        """

        """
            Device setting
        """
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        self.device = torch.device(device)

        """
            Communication setting
        """

        if compression_method == "dgc":
            from flock_grad_attack.compressor.dgc import (
                DgcCompressor,
            )

            self.compressor = DgcCompressor(compress_ratio)

        elif compression_method == "qsgd":
            from flock_grad_attack.compressor.qsgd import (
                QSGDCompressor,
            )

            self.compressor = QSGDCompressor(compress_ratio)

        elif compression_method == "topk":
            from flock_grad_attack.compressor.topk import (
                TopKCompressor,
            )

            self.compressor = TopKCompressor(compress_ratio)

        else:
            raise NotImplementedError(
                f"Not implemented compressor {compression_method}"
            )

    def get_starting_model(self):
        return CreditFraudNetMLP(num_features=self.features, num_classes=1)

    def process_dataset(self, dataset: list[dict], transform=None):
        logger.debug("Processing dataset")
        dataset_df = DataFrame.from_records(dataset)
        return get_loader(
            dataset_df, batch_size=batch_size, shuffle=True, drop_last=False
        )

    """
    train() should:
    1. Take in the model weights as bytes and load them into your model
    2. If parameters passed are None, initialize them to match your untrained model's parameters (i.e. clean slate)
    3. If needed pre-process the dataset which is passed as a list of rows parsed as dicts
    4. Output the model parameters retrained on the dataset AS BYTES
    """

    def train(self, parameters: bytes | None, dataset: list[dict]) -> bytes:
        data_loader = self.process_dataset(dataset)

        logger.info(f"--------- Trainer decompressing global model ---------")
        model = self.get_starting_model()
        if parameters is not None:
            model_compressed_dict = torch.load(io.BytesIO(parameters))
            for name, param in model.named_parameters():
                decompressed_tensor = self.compressor.decompress(
                    model_compressed_dict[name]["compressed_tensor"],
                    model_compressed_dict[name]["ctx"],
                )
                param.data = decompressed_tensor

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
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
                logger.info("INPUTS:")
                logger.info(inputs)
                logger.info("outputs:")
                logger.info(outputs)
                logger.info("targets:")
                logger.info(targets)

                loss = criterion(outputs, targets)
                loss.backward()

                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                predicted = torch.round(outputs).squeeze()
                train_total += targets.size(0)
                train_correct += (predicted == targets.squeeze()).sum().item()

            logger.info(
                f"Training Epoch: {epoch}, Acc: {round(100.0 * train_correct / train_total, 2)}, Loss: {round(train_loss / train_total, 4)}"
            )

        logger.info(f"--------- Trainer compressing local model ---------")
        model_compressed_dict = {}
        for name, param in model.named_parameters():
            compressed_tensor, ctx = self.compressor.compress(param, name)
            model_compressed_dict[name] = {
                "compressed_tensor": compressed_tensor,
                "ctx": ctx,
            }

        buffer = io.BytesIO()
        torch.save(model_compressed_dict, buffer)
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

        logger.info(f"--------- Evaluator decompressing global model ---------")
        model = self.get_starting_model()
        if parameters is not None:
            model_compressed_dict = torch.load(io.BytesIO(parameters))
            for name, param in model.named_parameters():
                decompressed_tensor = self.compressor.decompress(
                    model_compressed_dict[name]["compressed_tensor"],
                    model_compressed_dict[name]["ctx"],
                )
                param.data = decompressed_tensor

        model.to(self.device)
        model.eval()

        test_correct = 0
        test_loss = 0.0
        test_total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                logger.info("INPUTS:")
                logger.info(inputs)
                logger.info("outputs:")
                logger.info(outputs)
                logger.info("targets:")
                logger.info(targets)
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
        model = self.get_starting_model()

        logger.info(f"--------- Aggregator decompressing aggregated model ---------")
        decompressed_parameters_list = []
        for params in parameters_list:
            model_compressed_dict = torch.load(io.BytesIO(params))
            for name, param in model.named_parameters():
                decompressed_tensor = self.compressor.decompress(
                    model_compressed_dict[name]["compressed_tensor"],
                    model_compressed_dict[name]["ctx"],
                )
                param.data = decompressed_tensor
            decompressed_parameters_list.append(copy.deepcopy(model.state_dict()))

        averaged_params_template = decompressed_parameters_list[0]
        for k in averaged_params_template.keys():
            temp_w = []
            for local_w in decompressed_parameters_list:
                temp_w.append(local_w[k])
            averaged_params_template[k] = sum(temp_w) / len(temp_w)

        model.load_state_dict(averaged_params_template)

        logger.info(f"--------- Aggregator compressing aggregated model ---------")
        model_compressed_dict = {}
        for name, param in model.named_parameters():
            compressed_tensor, ctx = self.compressor.compress(param, name)
            model_compressed_dict[name] = {
                "compressed_tensor": compressed_tensor,
                "ctx": ctx,
            }

        # Create a buffer
        buffer = io.BytesIO()

        # Save state dict to the buffer
        torch.save(model_compressed_dict, buffer)

        # Get the byte representation
        aggregated_parameters = buffer.getvalue()

        return aggregated_parameters


def single_ml():
    """
    Hyper parameters
    """
    batch_size = 128
    epochs = 100
    lr = 0.0001
    # device = torch.device(f"cuda:{1 if torch.cuda.is_available() else 'cpu'}")
    device = torch.device("cpu")

    # for test
    client_id = 1

    # Data preparation for test
    from data_preprocessing import split_dataset, save_dataset

    datasets = split_dataset("data/creditcard.csv", num_clients=50, test_rate=0.2)
    save_dataset(datasets, "data")

    train_set = load_dataset(f"data/train_{client_id}.csv")
    test_set = load_dataset(f"data/test_{client_id}.csv")

    train_loader = get_loader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=False
    )
    test_loader = get_loader(
        test_set, batch_size=batch_size, shuffle=True, drop_last=False
    )

    """
        Create model
    """
    # a column for label
    model = CreditFraudNetMLP(num_features=train_set.shape[1] - 1, num_classes=1)

    """
        Single client training
    """

    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()

    pro_bar = tqdm(range(epochs))

    for _, epoch in enumerate(pro_bar):
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            predicted = torch.round(outputs).squeeze()
            train_total += targets.size(0)
            train_correct += (predicted == targets.squeeze()).sum().item()

        pro_bar.set_description(
            f"Training Epoch: {epoch}, Acc: {round(100.0 * train_correct / train_total, 2)}, Loss: {round(train_loss / train_total, 4)}"
        )

        if epochs % 10 == 0:
            model.eval()
            test_correct = 0
            test_loss = 0.0
            test_total = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(test_loader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    test_loss += loss.item() * inputs.size(0)
                    predicted = torch.round(outputs).squeeze()
                    test_total += targets.size(0)
                    test_correct += (predicted == targets.squeeze()).sum().item()

            logger.info(
                f"Test Epoch: {epoch}, Acc: {round(100.0 * test_correct / test_total, 2)}, Loss: {round(test_loss / test_total, 4)}"
            )
            model.train()


if __name__ == "__main__":
    """
    Hyper parameters
    """
    features = 30
    batch_size = 128
    epochs = 100
    lr = 0.0001
    classes = [
        "0",
        "1",
    ]

    from data_preprocessing import split_dataset, save_dataset

    datasets = split_dataset("data/creditcard.csv", num_clients=50, test_rate=0.2)
    save_dataset(datasets, "data")

    flock_model = FlockModel(
        classes,
        features,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
    )
    flock.register_train(flock_model.train)
    flock.register_evaluate(flock_model.evaluate)
    flock.register_aggregate(flock_model.aggregate)
    flock.run()
