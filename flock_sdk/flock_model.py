from abc import ABC, abstractmethod


class FlockModel(ABC):
    @abstractmethod
    def init_dataset(self, dataset_path: str) -> None:
        pass

    """
    train() should:
    1. Take in the model parameters as bytes and load them into your model
    according to your ML framework.
    2. Output the model parameters retrained on the dataset as bytes (!)
    """
    @abstractmethod
    def train(self, parameters: bytes) -> bytes:
        pass

    """
    evaluate() should:
    1. Take in the model parameters as bytes and load them into your model
    according to your ML framework.
    2. Output the accuracy of the model parameters on the dataset as a float
    """
    @abstractmethod
    def evaluate(self, parameters: bytes) -> bytes:
        pass

    """
    aggregate() should:
    1. Take in a list of model weights (bytes)
    2. Aggregate them using avg
    3. Output the aggregated parameters as bytes.
    """
    @abstractmethod
    def aggregate(self, parameters_list: list[bytes]) -> bytes:
        pass
