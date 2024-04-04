import os
from loguru import logger

def test_mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    logger.info(f"All params num: {all_model_params}, trainable param num: {trainable_model_params}")
    return trainable_model_params