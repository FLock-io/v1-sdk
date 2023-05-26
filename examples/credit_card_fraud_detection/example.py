import torch
import io
from loguru import logger
from flock_sdk import FlockSDK
import copy
import sys

import torch.utils.data
import torch.utils.data.distributed
from data_preprocessing import load_dataset, get_loader
from models.basic_cnn import CreditFraudNetMLP
from tqdm import tqdm
from compresser.dgc import dgc
from lightning import Fabric

flock = FlockSDK()


class FlockModel:
    def __init__(self, classes, fabric_instance=None, image_size=84, batch_size=256, epochs=1, lr=0.03, client_id = 1):
        """
            Hyper parameters
        """
        self.image_size = image_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.classes = classes
        self.class_to_idx = {_class: idx for idx, _class in enumerate(self.classes)}
        self.lr = lr

        """
            Data prepare
        """
        # for test
        self.train_set = load_dataset(f'data/train_{client_id}.csv')
        self.test_set = load_dataset(f'data/test_{client_id}.csv')

        """
            Device setting
        """
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        self.device = torch.device(device)

        """
            Training setting
        """
        # self.fabric = fabric_instance
        self.model = CreditFraudNetMLP(num_features=self.train_set.shape[1]-1, num_classes=1)

    def process_dataset(self, dataset: list[dict], transform=None):
        logger.debug("Processing dataset")
        return get_loader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    """
    train() should:
    1. Take in the model weights as bytes and load them into your model
    2. If parameters passed are None, initialize them to match your untrained model's parameters (i.e. clean slate)
    3. If needed pre-process the dataset which is passed as a list of rows parsed as dicts
    4. Output the model parameters retrained on the dataset AS BYTES
    """

    def train(self, compressed_gradients: bytes | None, dataset: list[dict]) -> bytes:
        data_loader = self.process_dataset(self.train_set)
        # data_loader = self.fabric.setup_dataloader(data_loader)

        if compressed_gradients is not None:
            compressed_grads = torch.load(io.BytesIO(compressed_gradients))
            for p, compressed_grad in zip(self.model.parameters(), compressed_grads):
                p.data.add_(compressed_grad)

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = torch.nn.BCELoss()
        self.model.to(self.device)

        uncompressed_buffer = io.BytesIO()
        torch.save(self.model.state_dict(), uncompressed_buffer)
        uncompressed_payload = uncompressed_buffer.getvalue()


        for epoch in range(self.epochs):
            logger.debug(f"Epoch {epoch}")
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                optimizer.zero_grad()

                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)

                loss = criterion(outputs, targets)
                loss.backward()
                # self.fabric.backward(loss)

                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                predicted = torch.round(outputs).squeeze()
                train_total += targets.size(0)
                train_correct += (predicted == targets.squeeze()).sum().item()
                if batch_idx < 2:
                    logger.debug(f"Batch {batch_idx}, Acc: {round(100.0 * train_correct / train_total, 2)}, Loss: {round(train_loss / train_total, 4)}")

            logger.info(
                f"Training Epoch: {epoch}, Acc: {round(100.0 * train_correct / train_total, 2)}, Loss: {round(train_loss / train_total, 4)}"
            )

        grads = [p.grad for p in self.model.parameters()]
        compressed_grads = dgc(grads)

        compressed_buffer = io.BytesIO()
        torch.save(compressed_grads, compressed_buffer)
        # return buffer.getvalue()
        compressed_payload = compressed_buffer.getvalue()
        logger.info(
            f"Delta Compression size: {self.payload_size_reformat(len(uncompressed_payload) - len(compressed_payload))}, "
            f"compressed ratio: {round((len(uncompressed_payload) - len(compressed_payload)) / len(uncompressed_payload) * 100, 2)}%, "
            f"original size: {self.payload_size_reformat(len(uncompressed_payload))}, "
            f"compressed size: {self.payload_size_reformat(len(compressed_payload))}")
        return compressed_payload

    """
    evaluate() should:
    1. Take in the model weights as bytes and load them into your model
    3. If parameters passed are None, initialize them to match your untrained model's parameters (i.e. clean slate)
    4. If needed pre-process the dataset which is passed as a list of rows parsed as dicts
    5. Output the accuracy of the model parameters on the dataset as a float
    """

    def evaluate(self, compressed_gradients: bytes | None, dataset: list[dict]) -> float:
        data_loader = self.process_dataset(self.test_set)
        criterion = torch.nn.BCELoss()

        if compressed_gradients is not None:
            compressed_grads = torch.load(io.BytesIO(compressed_gradients))
            for p, compressed_grad in zip(self.model.parameters(), compressed_grads):
                p.data.add_(compressed_grad)

        self.model.to(self.device)
        self.model.eval()

        test_correct = 0
        test_loss = 0.0
        test_total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item() * inputs.size(0)
                predicted = torch.round(outputs).squeeze()
                test_total += targets.size(0)
                test_correct += (predicted == targets.squeeze()).sum().item()
        accuracy = round(100.0 * test_correct / test_total, 2)
        logger.info(f"Model test, Acc: {accuracy}, Loss: {round(test_loss / test_total, 4)}")
        return accuracy

    """
    aggregate() should take in a list of model weights (bytes),
    aggregate them using avg and output the aggregated parameters as bytes.
    """

    def aggregate(self, compressed_gradients_list: list[bytes]) -> bytes:

        gradients_list = [torch.load(io.BytesIO(compressed_grads_bytes)) for compressed_grads_bytes in
                          compressed_gradients_list]

        # logger.info(f"len gradients_list {len(gradients_list)}")
        # logger.info(f"gradients_list : {gradients_list}")
        #
        # logger.info(f"gradients_list[0][0].shape : {gradients_list[0][0].shape}")
        # logger.info(f"gradients_list[1][0].shape : {gradients_list[1][0].shape}")

        transposed_gradients_list = list(map(list, zip(*gradients_list)))

        averaged_gradients = [torch.stack(tensors).mean(dim=0) for tensors in transposed_gradients_list]

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
            return f'{round(payload,3)} b'
        elif int(payload / 1024 / 1024) == 0:
            return f'{round(payload / 1024,3)} Kb'
        elif int(payload / 1024 / 1024 / 1024) == 0:
            return f'{round(payload / 1024 / 1024,3)} Mb'
        else:
            return f'{round(payload / 1024 / 1024 / 1024,3)} Gb'

if __name__ == "__main__":
    """
    Hyper parameters
    """
    batch_size = 128
    epochs = 100
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
        # fabric_instance=fabric,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
    )

    flock.register_train(flock_model.train)
    flock.register_evaluate(flock_model.evaluate)
    flock.register_aggregate(flock_model.aggregate)
    flock.run()
