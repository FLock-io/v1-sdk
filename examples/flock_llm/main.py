import os

import torch
import numpy as np
import random

from FLockLLMFinetuneModel import FLockLLMFinetuneModel
from arguments import load_arguments
from flock_sdk import FlockSDK
from loguru import logger
from s3_storage_manager import S3StorageManager, S3_MODEL_IMAGES_BUCKET
from utils.helper import test_mkdir
from utils.file_operations import extract_file

FLOCK_PRETRAINED_MODEL_HASH_DICT = {
    "google/gemma-2b" : "e383f1b83871aff02d3944ae6a5bfc88ae2068a75bfac32a2a04236c5bca0936",
    "mistralai/Mistral-7B-v0.1" : "c49ac9deff687228f52d0a7edf385a31b971ee81edd96994b9ffa9122aedde31",
}

def init(args):
    if args.mode == "experiment":
        init_random_seed(args.random_seed)

    if args.report_to == "wandb":
        if not os.getenv("WANDB_API_KEY") or os.getenv("WANDB_API_KEY") == "your_wandb_api_key_here":
            # raise ValueError("Please set your WANDB_API_KEY in the environment variables")
            logger.warning("Please set your WANDB_API_KEY of the environment variables in Dockerfile, now choose not logging to wandb")
            args.report_to = None

def prepare_pretrained_model(args):
    if args.foundation_model_pre_trained_weights_source == "huggingface":
        args.foundation_model_pre_trained_weights_path = args.foundation_model
    elif args.foundation_model_pre_trained_weights_source == "flock_s3":
        s3 = S3StorageManager(S3_MODEL_IMAGES_BUCKET)
        if args.foundation_model in FLOCK_PRETRAINED_MODEL_HASH_DICT:
            _hash = FLOCK_PRETRAINED_MODEL_HASH_DICT[args.foundation_model]
            download_path = "models"
            test_mkdir(download_path)
            target_download_file_path = os.path.join(download_path, _hash)
            if not os.path.exists(target_download_file_path):
                s3.download_file(_hash=_hash, directory_path=download_path)
            else:
                logger.warning(
                    f"The foundation model {args.foundation_model} with hash {_hash} file is downloaded in {target_download_file_path}")
            # Decompress the file
            extract_file(target_download_file_path, download_path, args.foundation_model.split("/")[-1])
            args.foundation_model_pre_trained_weights_path = os.path.join(download_path, args.foundation_model.split("/")[-1])
        else:
            raise ValueError(f"Invalid foundation_model {args.foundation_model} in FLock S3 storage")
    else:
        raise ValueError(f"Invalid foundation_model_pre_trained_weights_source {args.foundation_model_pre_trained_weights_source}")


def init_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":

    args = load_arguments()

    init(args)

    prepare_pretrained_model(args)

    llm_finetune_model = FLockLLMFinetuneModel(args)

    sdk = FlockSDK(llm_finetune_model)
    sdk.run()

    ################################################################################
    # Test only
    ################################################################################
    # w = llm_finetune_model.train(parameters=None)
    # import copy
    #
    # agg_w = llm_finetune_model.aggregate(parameters_list=[w, copy.deepcopy(w)])
    # final_loss = llm_finetune_model.evaluate(parameters=agg_w)
    # #
    # print(f"Final loss: {final_loss}")
