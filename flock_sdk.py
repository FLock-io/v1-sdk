import base64

from flask import jsonify, Flask, request
import logging


class FlockSDK:
    def __init__(self):
        self.flask = Flask(__name__)
        self.methods = {}
        self.debug = debug
        logging.basicConfig(
            level=logging.DEBUG if self.debug else logging.INFO)

    def _register_view(self, name, func):
        self.methods[name] = func
        self.flask.add_url_rule(
            f"/{name}",
            endpoint=name,
            view_func=func,
            methods=["POST", "OPTIONS"],
        )

    def evaluate(self, func):
        def wrapper(*args, **kwargs):
            data = request.get_json(force=True)

            if not isinstance(data, dict):
                logging.error("Invalid data: expected a dictionary.")
                raise ValueError("Invalid data: expected a dictionary.")
            if "parameters" not in data or "dataset" not in data:
                logging.error(
                    "Missing required keys in data: 'parameters' and 'dataset' are required.")
                raise ValueError(
                    "Missing required keys in data: 'parameters' and 'dataset' are required.")

            parameters = base64.b64decode(data["parameters"])
            dataset = data["dataset"]
            accuracy = func(parameters, dataset)
            return jsonify({"accuracy": accuracy})

        self._register_view("evaluate", wrapper)
        return wrapper

    def train(self, func):
        def wrapper(*args, **kwargs):
            data = request.get_json(force=True)
            parameters = base64.b64decode(data["parameters"])
            dataset = data["dataset"]
            trained_parameters = func(parameters, dataset)
            b64_parameters = base64.b64encode(trained_parameters)
            return jsonify({"parameters": b64_parameters.decode("ascii")})

        self._register_view("train", wrapper)
        return wrapper

    def aggregate(self, func):
        def wrapper(*args, **kwargs):
            data = request.get_json(force=True)
            parameters_list = [
                base64.b64decode(parameters) for paramters in data["parameters_list"]
            ]
            aggregated_parameters = func(parameters_list)
            b64_parameters = base64.b64encode(aggregated_parameters)
            return jsonify({"parameters": b64_parameters.decode("ascii")})

        self._register_view("aggregate", wrapper)
        return func

    def _check_registered_methods(self):
        assert set(self.methods.keys()) == set(
            ["aggregate", "evaluate", "train"])

    def run(self):
        self._check_registered_methods()
        self.flask.run(debug=self.debug)
