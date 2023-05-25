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

    def train(self, parameters: bytes | None, dataset: list[dict]) -> bytes:
        data_loader = self.process_dataset(self.train_set)
        # data_loader = self.fabric.setup_dataloader(data_loader)

        if parameters is not None:
            self.model.load_state_dict(torch.load(io.BytesIO(parameters)))

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = torch.nn.BCELoss()
        self.model.to(self.device)

        # size_before_compression = 0
        # for p in self.model.parameters():
        #     if p.grad is not None:
        #         size_before_compression += p.grad.element_size() * p.grad.nelement()
        #     else:
        #         logger.info("None grad")

        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        uncompressed_payload = buffer.getvalue()


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

                # Compress gradients
                grads = [p.grad for p in self.model.parameters()]
                compressed_grads = dgc(grads)
                # Manually update model parameters
                for p, compressed_grad in zip(self.model.parameters(), compressed_grads):
                    p.data.add_(-self.lr, compressed_grad)

                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                predicted = torch.round(outputs).squeeze()
                train_total += targets.size(0)
                train_correct += (predicted == targets.squeeze()).sum().item()

            logger.info(
                f"Training Epoch: {epoch}, Acc: {round(100.0 * train_correct / train_total, 2)}, Loss: {round(train_loss / train_total, 4)}"
            )

        # size_after_compression = 0
        # for p in self.model.parameters():
        #     if p.grad is not None:
        #         size_after_compression += p.grad.element_size() * p.grad.nelement()
        #     else:
        #         logger.info("None grad")
        # delta_compression_size = size_before_compression - size_after_compression
        # logger.info(f"Delta Compression size: {self.payload_size_reformat(delta_compression_size)}, "
        #             f"compressed ratio: {round(delta_compression_size / size_before_compression * 100,2) if round(delta_compression_size / size_before_compression * 100,2) !=0 else 0}%, "
        #             f"original size: {self.payload_size_reformat(size_before_compression)}, "
        #             f"compressed size: {self.payload_size_reformat(size_after_compression)}")

        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        # return buffer.getvalue()
        compressed_payload = buffer.getvalue()
        logger.info(f"Delta Compression size: {self.payload_size_reformat(sys.getsizeof(uncompressed_payload) -  sys.getsizeof(compressed_payload))}, "
                    f"compressed ratio: {round((sys.getsizeof(uncompressed_payload) -  sys.getsizeof(compressed_payload)) / sys.getsizeof(uncompressed_payload) * 100,2)}%, "
                    f"original size: {self.payload_size_reformat(uncompressed_payload)}, "
                    f"compressed size: {self.payload_size_reformat(compressed_payload)}")

        return compressed_payload

    """
    evaluate() should:
    1. Take in the model weights as bytes and load them into your model
    3. If parameters passed are None, initialize them to match your untrained model's parameters (i.e. clean slate)
    4. If needed pre-process the dataset which is passed as a list of rows parsed as dicts
    5. Output the accuracy of the model parameters on the dataset as a float
    """

    def evaluate(self, parameters: bytes | None, dataset: list[dict]) -> float:
        data_loader = self.process_dataset(self.test_set)
        criterion = torch.nn.BCELoss()

        if parameters is not None:
            self.model.load_state_dict(torch.load(io.BytesIO(parameters)))
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

    def payload_size_reformat(self, payload):
        # Monitor size of model
        if int(sys.getsizeof(payload) / 1024) == 0:
            return f'{round(sys.getsizeof(payload),3)} b'
        elif int(sys.getsizeof(payload) / 1024 / 1024) == 0:
            return f'{round(sys.getsizeof(payload) / 1024,3)} Kb'
        elif int(sys.getsizeof(payload) / 1024 / 1024 / 1024) == 0:
            return f'{round(sys.getsizeof(payload) / 1024 / 1024,3)} Mb'
        else:
            return f'{round(sys.getsizeof(payload) / 1024 / 1024 / 1024,3)} Gb'


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
