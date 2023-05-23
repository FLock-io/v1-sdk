import base64
from flask import jsonify
from flask import Flask
from flask import request
from loguru import logger

from functools import wraps


class FlockSDK:
    def __init__(self, debug: bool = True):
        self.flask = Flask(__name__)
        self.methods = {}
        self.debug = debug

    def _register_view(self, name, func):
        self.methods[name] = func
        self.flask.add_url_rule(
            f"/{name}",
            endpoint=name,
            view_func=func,
            methods=["POST", "OPTIONS"],
        )

    def register_evaluate(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            data = request.get_json(force=True)
            parameters = data["parameters"]
            if parameters:
                parameters = base64.b64decode(data["parameters"])

            dataset = data["dataset"]
            accuracy = func(parameters, dataset)
            return jsonify({"accuracy": accuracy})

        self._register_view("evaluate", wrapper)
        return wrapper

    def register_train(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            data = request.get_json(force=True)
            parameters = data["parameters"]
            if parameters:
                parameters = base64.b64decode(data["parameters"])

            dataset = data["dataset"]
            trained_parameters = func(parameters, dataset)
            b64_parameters = base64.b64encode(trained_parameters)
            return jsonify({"parameters": b64_parameters.decode("ascii")})

        self._register_view("train", wrapper)
        return wrapper

    def register_aggregate(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            data = request.get_json(force=True)
            parameters_list = [
                base64.b64decode(parameters) for parameters in data["parameters_list"]
            ]
            aggregated_parameters = func(parameters_list)
            b64_parameters = base64.b64encode(aggregated_parameters)
            return jsonify({"parameters": b64_parameters.decode("ascii")})

        self._register_view("aggregate", wrapper)
        return func

    def _check_registered_methods(self):
        assert set(self.methods.keys()) == set(["aggregate", "evaluate", "train"])

    def run(self):
        self._check_registered_methods()
        self.flask.run(host="0.0.0.0", debug=self.debug)
