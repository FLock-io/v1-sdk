from typing import Any
from flock_sdk import FlockSDK

flock = FlockSDK()


@flock.evaluate
def evaluate(parameters: bytes, dataset: list[bytes]) -> float:
    return 0.95


@flock.train
def train(parameters: bytes, dataset: list[bytes]) -> bytes:
    new_parameters = b""
    return new_parameters


@flock.aggregate
def aggregate(parameters_list: list[bytes]) -> bytes:
    aggregated_parameters = b""
    return aggregated_parameters


if __name__ == "__main__":
    flock.run()
