import torch
import numpy as np
import random

from FLockLLMFinetuneModel import FLockLLMFinetuneModel
from arguments import load_arguments
from flock_sdk import FlockSDK


def init(args):
    if args.mode == "experiment":
        init_random_seed(args.random_seed)


def init_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":

    args = load_arguments()

    init(args)

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
