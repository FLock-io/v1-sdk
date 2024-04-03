'''
    Ref FedML: https://github.com/FedML-AI/FedML/blob/master/python/fedml/arguments.py
'''

import argparse
from os import path

import yaml

def add_args():
    parser = argparse.ArgumentParser(description="FedContinuum")
    parser.add_argument(
        "--yaml_config_file",
        "--conf",
        help="configuration file in yaml",
        type=str,
        default="",
    )

    args, unknown = parser.parse_known_args()
    return args

class Arguments:

    def __init__(self, cmd_args, override_cmd_args=True):
        # set the command line arguments
        cmd_args_dict = cmd_args.__dict__
        for arg_key, arg_val in cmd_args_dict.items():
            setattr(self, arg_key, arg_val)

        self.get_default_yaml_config(cmd_args)
        if not override_cmd_args:
            # reload cmd args again
            for arg_key, arg_val in cmd_args_dict.items():
                setattr(self, arg_key, arg_val)

    def load_yaml_config(self, yaml_path):
        try:
            with open(yaml_path, "r") as stream:
                try:
                    return yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    raise ValueError("Yaml error - check yaml file")
        except Exception as e:
            return None

    def get_default_yaml_config(self, cmd_args):
        if cmd_args.yaml_config_file == "":
            path_current_file = path.abspath(path.dirname(__file__))
            raise Exception(f"yaml_config_file is not specified or cannot fined via {path_current_file}")

        self.yaml_paths = [cmd_args.yaml_config_file]
        # Load all arguments from yaml config
        # https://www.cloudbees.com/blog/yaml-tutorial-everything-you-need-get-started
        configuration = self.load_yaml_config(cmd_args.yaml_config_file)

        # Override class attributes from current yaml config
        if configuration is not None:
            self.set_attr_from_config(configuration)

        return configuration

    def set_attr_from_config(self, configuration):
        for _, param_family in configuration.items():
            for key, val in param_family.items():
                setattr(self, key, val)


def load_arguments():
    cmd_args = add_args()
    # Load all arguments from YAML config file
    args = Arguments(cmd_args)

    return args