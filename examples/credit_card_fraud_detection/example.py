import io

import torch
import torch.utils.data
import torch.utils.data.distributed
from loguru import logger
# from lightning import Fabric
from pandas import DataFrame

from compresser.dgc import dgc
from data_preprocessing import get_loader
from flock_sdk import FlockSDK
from models.basic_cnn import CreditFraudNetMLP

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
            Data prepare
        """
        # for test
        # self.train_set = load_dataset(f"data/train_{client_id}.csv")
        # self.test_set = load_dataset(f"data/test_{client_id}.csv")

        """
            Device setting
        """
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        self.device = torch.device("cpu")

        """
            Training setting
        """
        # self.fabric = fabric_instance

    def process_dataset(self, dataset: list[dict], transform=None):
        logger.debug("Processing dataset")
        dataset_df = DataFrame.from_records(dataset)
        return get_loader(
            dataset_df, batch_size=batch_size, shuffle=True, drop_last=False
        )

    def get_starting_model(self):
        torch.manual_seed(0)
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
        # data_loader = self.fabric.setup_dataloader(data_loader)

        model = self.get_starting_model()

        if parameters is not None:
            compressed_grads = torch.load(io.BytesIO(parameters))
            for p, compressed_grad in zip(model.parameters(), compressed_grads):
                p.data.add_(-self.lr * compressed_grad)

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = torch.nn.BCELoss()
        model.to(self.device)

        for epoch in range(self.epochs):
            logger.debug(f"Epoch {epoch}")
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                #                 optimizer.zero_grad()

                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)

                loss = criterion(outputs, targets)
                loss.backward()
                # self.fabric.backward(loss)

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

        uncompressed_grads = [p.grad for p in model.parameters()]
        uncompressed_buffer = io.BytesIO()
        torch.save(uncompressed_grads, uncompressed_buffer)
        uncompressed_payload = uncompressed_buffer.getvalue()
        # Perform DGC
        compressed_grads = dgc(uncompressed_grads)
        compressed_buffer = io.BytesIO()
        torch.save(compressed_grads, compressed_buffer)
        # return buffer.getvalue()
        compressed_payload = compressed_buffer.getvalue()

        logger.info(
            f"Delta Compression size: {self.payload_size_reformat(len(uncompressed_payload) - len(compressed_payload))}, "
            f"compressed ratio: {round((len(uncompressed_payload) - len(compressed_payload)) / len(uncompressed_payload) * 100, 2)}%, "
            f"original size: {self.payload_size_reformat(len(uncompressed_payload))}, "
            f"compressed size: {self.payload_size_reformat(len(compressed_payload))}"
        )
        return compressed_payload

    """
    evaluate() should:
    1. Take in the model weights as bytes and load them into your model
    3. If parameters passed are None, initialize them to match your untrained model's parameters (i.e. clean slate)
    4. If needed pre-process the dataset which is passed as a list of rows parsed as dicts
    5. Output the accuracy of the model parameters on the dataset as a float
    """

    def evaluate(
            self, compressed_gradients: bytes | None, dataset: list[dict]
    ) -> float:
        data_loader = self.process_dataset(dataset)
        model = self.get_starting_model()
        criterion = torch.nn.BCELoss()

        if compressed_gradients is not None:
            compressed_grads = torch.load(io.BytesIO(compressed_gradients))
            for p, compressed_grad in zip(model.parameters(), compressed_grads):
                p.data.add_(-self.lr * compressed_grad)

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
        gradients_list = [
            torch.load(io.BytesIO(compressed_grads_bytes))
            for compressed_grads_bytes in parameters_list
        ]

        transposed_gradients_list = list(map(list, zip(*gradients_list)))
        # Load with dense tensor.
        transposed_gradients_list = list(map(lambda compressed_tensor: compressed_tensor.to_dense(),
                                             transposed_gradients_list))

        averaged_gradients = [
            torch.stack(tensors).mean(dim=0) for tensors in transposed_gradients_list
        ]

        # Create a buffer
        buffer = io.BytesIO()

        # Save state dict to the buffer
        torch.save(averaged_gradients, buffer)

        # Get the byte representation
        aggregated_parameters = buffer.getvalue()

        return aggregated_parameters

    def payload_size_reformat(self, payload):
        # Monitor size of model
        if int(payload / 1024) == 0:
            return f"{round(payload, 3)} b"
        elif int(payload / 1024 / 1024) == 0:
            return f"{round(payload / 1024, 3)} Kb"
        elif int(payload / 1024 / 1024 / 1024) == 0:
            return f"{round(payload / 1024 / 1024, 3)} Mb"
        else:
            return f"{round(payload / 1024 / 1024 / 1024, 3)} Gb"


if __name__ == "__main__":
    """
    Hyper parameters
    """
    batch_size = 128
    epochs = 1
    lr = 0.0001
    classes = [
        "0",
        "1",
    ]

    # # Add Fabric support
    # fabric = Fabric(accelerator="auto", devices=-1, strategy="ddp")
    # fabric.launch()

    flock_model = FlockModel(
        classes,
        30,
        # fabric_instance=fabric,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        output_num_classes=1,
    )

    import json

    with open("test_dataset.json", "r") as f:
        dataset = json.loads(f.read())

    compressed_gradient = flock_model.train(None, dataset)
    flock_model.evaluate(compressed_gradient, dataset)
    # flock.register_train(flock_model.train)
    # flock.register_evaluate(flock_model.evaluate)
    # flock.register_aggregate(flock_model.aggregate)
    # flock.run()
