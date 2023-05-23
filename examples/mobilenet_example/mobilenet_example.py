import copy
import torch
from tqdm import tqdm
import io
from loguru import logger
from base64 import b64decode
from PIL import Image
from flock_sdk import FlockSDK
from mobilenet_v3 import MobileNetV3

flock = FlockSDK()


class FlockModel:
    def __init__(self, classes, image_size=84, batch_size=256, epochs=1, lr=0.03):
        """
        Hyper parameters
        """
        self.image_size = image_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.classes = classes
        self.class_to_idx = {_class: idx for idx, _class in self.classes}
        self.lr = lr
        """
            Data prepare
        """
        self.transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.transform_test = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        """
            Device setting
        """
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        self.device = torch.device(device)

    def get_starting_model(self):
        return MobileNetV3(
            model_mode="LARGE",
            num_classes=len(self.classes),
            multiplier=1.0,
            dropout_rate=0.0,
        )

    def process_dataset(self, dataset: list[dict], transform=None):
        logger.debug("Processing dataset")
        for idx, row in enumerate(dataset):
            dataset[idx]["label"] = self.class_to_idx[row["label"]]

        dataset = [
            (
                Image.open(io.BytesIO(b64decode(row["image"]))).convert("RGB"),
                torch.tensor(row["label"]),
            )
            for row in dataset
        ]
        if transform:
            dataset = [(transform(row[0]), row[1]) for row in dataset]
        logger.debug("Dataset processed")
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

    """
    train() should:
    1. Take in the model weights as bytes and load them into your model
    2. If parameters passed are None, initialize them to match your untrained model's parameters (i.e. clean slate)
    3. If needed pre-process the dataset which is passed as a list of rows parsed as dicts
    4. Output the model parameters retrained on the dataset AS BYTES
    """

    def train(self, parameters: bytes | None, dataset: list[dict]) -> bytes:
        data_loader = self.process_dataset(dataset, self.transform_train)

        model = self.get_starting_model()
        if parameters is not None:
            model.load_state_dict(torch.load(io.BytesIO(parameters)))
        model.train()
        opti = torch.optim.SGD(
            model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4
        )
        loss_func = torch.nn.CrossEntropyLoss()
        model.to(self.device)

        # pro_bar = tqdm(range(self.epochs))
        for epoch in range(self.epochs):
            logger.debug(f"Epoch {epoch}")
            batch_loss = []
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                # logger.info(f"Inputs: {inputs}")
                # logger.info(f"Targets: {targets}")
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                # logger.info(f"Outputs: {outputs}")
                loss = loss_func(outputs, targets)
                opti.zero_grad()
                loss.backward()
                opti.step()
                batch_loss.append(loss.item())
                logger.info(f"Batch idx: {batch_idx}")

            logger.info(
                f"Epoch: {epoch}, Loss: {round(sum(batch_loss) / len(batch_loss), 4)}"
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
        data_loader = self.process_dataset(dataset, self.transform_test)

        model = self.get_starting_model()
        if parameters is not None:
            model.load_state_dict(torch.load(io.BytesIO(parameters)))
        model.to(self.device)
        model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        accuracy = correct / total
        logger.info(f"Accuracy: {accuracy}")
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


import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

if __name__ == "__main__":
    """
    Hyper parameters
    """
    image_size = 84
    batch_size = 256
    epochs = 5
    lr = 0.1
    classes = [
        "n02006656",
        "n02096585",
        "n02105641",
        "n02268853",
        "n02325366",
        "n02791270",
        "n02814533",
        "n03180011",
        "n03788195",
        "n04525038",
        "n04509417",
        "n02641379",
        "n02804414",
        "n03691459",
        "n07749582",
        "n10565667",
        "n12620546",
        "n02488702",
        "n03417042",
        "n03794056",
    ]

    flock_model = FlockModel(
        classes,
        image_size=image_size,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
    )
    flock.register_train(flock_model.train)
    flock.register_evaluate(flock_model.evaluate)
    flock.register_aggregate(flock_model.aggregate)
    flock.run()
