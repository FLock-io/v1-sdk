from flock_sdk import FlockSDK

flock = FlockSDK()


"""
evaluate() should:
1. Take in the model weights as bytes and load them into your model
2. If needed pre-process the dataset which is passed as a list of rows parsed as dicts
3. Output the accuracy of the model parameters on the dataset as a float
"""


@flock.evaluate
def evaluate(parameters: bytes, dataset: list[dict]) -> float:
    return 0.95


"""
train() should:
1. Take in the model weights as bytes and load them into your model
2. If needed pre-process the dataset which is passed as a list of rows parsed as dicts
3. Output the model parameters retrained on the dataset AS BYTES
"""


@flock.train
def train(parameters: bytes, dataset: list[dict]) -> bytes:
    new_parameters = b""
    return new_parameters


"""
aggregate() should take in a list of model weights (bytes),
aggregate them using avg and output the aggregated parameters as bytes.
"""


@flock.aggregate
def aggregate(parameters_list: list[bytes]) -> bytes:
    aggregated_parameters = b""
    return aggregated_parameters


if __name__ == "__main__":
    flock.run()
