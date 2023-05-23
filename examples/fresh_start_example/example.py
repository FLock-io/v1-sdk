from flock_sdk import FlockSDK

flock = FlockSDK()


class FlockModel:
    def get_starting_parameters(self) -> bytes:
        return b""

    """
    train() should:
    1. Take in the model weights as bytes and load them into your model
    2. If parameters passed are None, initialize them to match your untrained model's parameters (i.e. clean slate)
    3. If needed pre-process the dataset which is passed as a list of rows parsed as dicts
    4. Output the model parameters retrained on the dataset AS BYTES
    """

    def train(self, parameters: bytes | None, dataset: list[dict]) -> bytes:
        if parameters == None:
            parameters = self.get_starting_parameters()
        new_parameters = b""
        return new_parameters

    """
    evaluate() should:
    1. Take in the model weights as bytes and load them into your model
    3. If parameters passed are None, initialize them to match your untrained model's parameters (i.e. clean slate)
    4. If needed pre-process the dataset which is passed as a list of rows parsed as dicts
    5. Output the accuracy of the model parameters on the dataset as a float
    """

    def evaluate(self, parameters: bytes | None, dataset: list[dict]) -> float:
        if parameters == None:
            parameters = self.get_starting_parameters()
        accuracy = 0.25
        return accuracy

    """
    aggregate() should take in a list of model weights (bytes),
    aggregate them using avg and output the aggregated parameters as bytes.
    """

    def aggregate(self, parameters_list: list[bytes]) -> bytes:
        aggregated_parameters = b""
        return aggregated_parameters


if __name__ == "__main__":
    model = FlockModel()
    flock.register_train(model.train)
    flock.register_evaluate(model.evaluate)
    flock.register_aggregate(model.aggregate)
    flock.run()
