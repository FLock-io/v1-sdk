import base64
from flask import jsonify
from flask import Flask
from flask import request
from loguru import logger

from functools import wraps

# from .flock_model import FlockModel


class FlockSDK:
    def __init__(self, model, debug: bool = True):
        self.flask = Flask(__name__)
        self.model = model
        self.debug = debug
        self.flask.add_url_rule(
            "/call",
            endpoint="call",
            view_func=self.call,
            methods=["POST", "OPTIONS"],
        )
        self.model.init_dataset("/dataset.json")

    def call(self, *args, **kwargs):
        data = request.get_json(force=True)
        method = data.pop("method")
        if method not in ["train", "evaluate", "aggregate"]:
            raise RuntimeError("Attempted to call invalid SDK method!")
        func = getattr(self.model, method)

        parameters = data.pop("parameters", None)
        if parameters:
            parameters = base64.b64decode(parameters)

        parameters_list = data.pop("parameters_list", None)
        if parameters_list:
            parameters_list = [
                base64.b64decode(parameters) for parameters in parameters_list
            ]

        if method == "train":
            trained_parameters = func(parameters)
            b64_parameters = base64.b64encode(trained_parameters)
            return jsonify({"parameters": b64_parameters.decode("ascii")})
        elif method == "evaluate":
            accuracy = func(parameters)
            return jsonify({"accuracy": accuracy})
        elif method == "aggregate":
            aggregated_parameters = func(parameters_list)
            b64_parameters = base64.b64encode(aggregated_parameters)
            return jsonify({"parameters": b64_parameters.decode("ascii")})
        else:
            raise ValueError("Incorrect arguments passed to SDK function")

    def run(self):
        self.flask.run(host="0.0.0.0", debug=self.debug)
