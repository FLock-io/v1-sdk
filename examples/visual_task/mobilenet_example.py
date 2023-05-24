import io
from base64 import b64decode

import torch
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import Image
from lightning import Fabric
from loguru import logger
from timm import create_model

from flock_sdk import FlockSDK

# Call FLock SDK.
flock = FlockSDK()


class FLockVisual:
    def __init__(self, model_name, classes, fabric_instance, lr=0.03):
        num_classes = len(classes)
        self.model = create_model(model_name, num_classes)
        self.batch_size = batch_size
        self.epochs = epochs
        self.classes = classes
        self.fabric = fabric_instance
        self.class_to_idx = {_class: idx for idx, _class in self.classes}
        self.lr = lr
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

    def train(self, parameters: None, dataset: list[dict]) -> bytes:
        data_loader = self.process_dataset(dataset, self.transform_train)
        data_loader = self.fabric.setup_dataloader(data_loader)
        if parameters is not None:
            self.model.load_state_dict(torch.load(io.BytesIO(parameters)))
        self.model.train()
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # pro_bar = tqdm(range(self.epochs))
        for epoch in range(self.epochs):
            logger.debug(f"Epoch {epoch}")
            batch_loss = []
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss_val = F.cross_entropy(outputs, targets)
                fabric.backward(loss_val)
                optimizer.step()
                batch_loss.append(loss_val.item())
                logger.info(f"Batch idx: {batch_idx}")
            logger.info(
                f"Epoch: {epoch}, Loss: {round(sum(batch_loss) / len(batch_loss), 4)}"
            )

        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        return buffer.getvalue()

    """
    evaluate() should:
    1. Take in the model weights as bytes and load them into your model
    3. If parameters passed are None, initialize them to match your untrained model's parameters (i.e. clean slate)
    4. If needed pre-process the dataset which is passed as a list of rows parsed as dicts
    5. Output the accuracy of the model parameters on the dataset as a float
    """

    def evaluate(self, parameters: None, dataset: list[dict]) -> float:
        data_loader = self.process_dataset(dataset, self.transform_test)

        if parameters is not None:
            self.model.load_state_dict(torch.load(io.BytesIO(parameters)))
        self.model.to(self.device)
        self.model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
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


if __name__ == "__main__":
    """
    Hyper parameters
    """
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

    # Add Fabric support
    fabric = Fabric(accelerator="cuda", devices=-1, strategy="ddp")
    fabric.launch()

    flock_model = FLockVisual(
        classes,
        fabric_instance=fabric,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
    )
    flock.register_train(flock_model.train)
    flock.register_evaluate(flock_model.evaluate)
    flock.register_aggregate(flock_model.aggregate)
    flock.run()
