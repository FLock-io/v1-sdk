import torch
import io
from loguru import logger
from flock_sdk.flock_sdk import FlockSDK

from pandas import DataFrame
import numpy as np
import random

flock = FlockSDK()


import os
from typing import List

from transformers import LlamaTokenizer, LlamaForCausalLM
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
    get_peft_model_state_dict,
)
from federatedgpt_shepherd.fed_utils import GeneralClient
import datasets
from datasets import load_dataset

from federatedgpt_shepherd.utils.prompter import Prompter

datasets.utils.logging.set_verbosity_error()

class FlockModel:
    def __init__(
            self,
            # model/data params
            global_model: str = '',
            data_path: str = './data',
            output_dir: str = './lora-shepherd/',
            # FL hyperparamas
            client_selection_frac: float = 0.1,
            num_communication_rounds: int = 50,
            num_clients: int = 10,
            # Local training hyperparams
            local_batch_size: int = 64,  # 64,
            local_micro_batch_size: int = 8,
            local_num_epochs: int = 10,
            local_learning_rate: float = 3e-4,
            local_val_set_size: int = 0,
            local_save_steps: int = 3,
            cutoff_len: int = 512,
            # LoRA hyperparams
            lora_r: int = 16,
            lora_alpha: int = 16,
            lora_dropout: float = 0.05,
            lora_target_modules: List[str] = [
                "q_proj",
            ],
            # llm hyperparams
            train_on_inputs: bool = True,
            group_by_length: bool = False,
            resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
            prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
            seed= 0,
            client_id=1,
    ):
        self.client_id = client_id
        self.local_comm_round_idx = 0

        """
            Environment variables
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

        """
        Hyper parameters
        """
        self.global_model = global_model
        self.data_path = data_path
        self.output_dir = output_dir
        self.client_selection_frac = client_selection_frac
        self.num_communication_rounds = num_communication_rounds
        self.num_clients = num_clients
        self.num_clients_per_round = int(num_clients * client_selection_frac) if int(num_clients * client_selection_frac) > 0 else 1
        self.local_batch_size = local_batch_size
        self.local_micro_batch_size = local_micro_batch_size
        self.local_num_epochs = local_num_epochs
        self.local_learning_rate = local_learning_rate
        self.local_val_set_size = local_val_set_size
        self.local_save_steps = local_save_steps
        self.cutoff_len = cutoff_len
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules
        self.train_on_inputs = train_on_inputs
        self.group_by_length = group_by_length
        self.resume_from_checkpoint = resume_from_checkpoint
        self.prompt_template_name = prompt_template_name

        logger.debug(
            f"Federated Finetuning LLM-LoRA with params:\n"
            f"global_model: {global_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"client_selection_frac: {client_selection_frac}\n"
            f"num_communication_rounds: {num_communication_rounds}\n"
            f"num_clients: {num_clients}\n"
            f"local_batch_size: {local_batch_size}\n"
            f"local_micro_batch_size: {local_micro_batch_size}\n"
            f"local_num_epochs: {local_num_epochs}\n"
            f"local_learning_rate: {local_learning_rate}\n"
            f"local_val_set_size: {local_val_set_size}\n"
            f"local_save_steps: {local_save_steps}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )

        """
            LLM Settings & Preparation
        """
        self.gradient_accumulation_steps = local_batch_size // local_micro_batch_size
        self.prompter = Prompter(prompt_template_name)
        self.device_map = "auto"
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.ddp = world_size != 1
        if self.ddp:
            self.device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
            self.gradient_accumulation_steps = self.gradient_accumulation_steps // world_size

        self.tokenizer = LlamaTokenizer.from_pretrained(global_model)
        self.tokenizer.pad_token_id = (
            0
        )
        self.tokenizer.padding_side = "left"

        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        """
            Device setting
        """
        # if torch.cuda.is_available():
        #     device = "cuda"
        # else:
        #     device = "cpu"
        # self.device = torch.device(device)


    def load_dataset(self, generate_and_tokenize_prompt, local_val_set_size):

        self.local_data_path = os.path.join(data_path, "local_training_{}.json".format(self.client_id))
        self.local_data = load_dataset("json", data_files=self.local_data_path)

        if local_val_set_size > 0:
            local_train_val = self.local_data["train"].train_test_split(
                test_size=local_val_set_size, shuffle=True, seed=42
            )
            self.local_train_dataset = (
                local_train_val["train"].shuffle().map(generate_and_tokenize_prompt)
            )
            self.local_eval_dataset = (
                local_train_val["test"].shuffle().map(generate_and_tokenize_prompt)
            )
        else:
            self.local_train_dataset = self.local_data["train"].shuffle().map(generate_and_tokenize_prompt)
            self.local_eval_dataset = None
        self.local_val_set_size = local_val_set_size

        return self.local_train_dataset, self.local_eval_dataset

    def tokenize(self, prompt, add_eos_token=True):
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != self.tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(self, data_point):
        full_prompt = self.prompter.generate_prompt(
            data_point["instruction"],
            data_point["context"],
            data_point["response"],
        )
        tokenized_full_prompt = self.tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = self.prompter.generate_prompt(
                data_point["instruction"], data_point["context"]
            )
            tokenized_user_prompt = self.tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt

    def get_starting_model(self):
        model = LlamaForCausalLM.from_pretrained(
            self.global_model,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map=self.device_map,
        )

        model = prepare_model_for_int8_training(model)

        model = get_peft_model(model, self.lora_config)
        if not self.ddp and torch.cuda.device_count() > 1:
            model.is_parallelizable = True
            model.model_parallel = True

        return model

    """
    train() should:
    1. Take in the model weights as bytes and load them into your model
    2. If parameters passed are None, initialize them to match your untrained model's parameters (i.e. clean slate)
    3. If needed pre-process the dataset which is passed as a list of rows parsed as dicts
    4. Output the model parameters retrained on the dataset AS BYTES
    """

    def train(self, parameters: bytes | None) -> bytes:
        self.local_comm_round_idx += 1

        logger.info("\nPreparing the local dataset and trainer")

        local_train_dataset, local_eval_dataset = self.load_dataset(self.generate_and_tokenize_prompt, local_val_set_size)

        model = self.get_starting_model()
        if parameters is not None:
            set_peft_model_state_dict(model, torch.load(io.BytesIO(parameters)), "default")

        model.train()
        client = GeneralClient(self.client_id, model, local_train_dataset, local_eval_dataset, self.local_val_set_size, self.output_dir)

        client.build_local_trainer(self.tokenizer,
                                   self.local_micro_batch_size,
                                   self.gradient_accumulation_steps,
                                   self.local_num_epochs,
                                   self.local_learning_rate,
                                   self.group_by_length,
                                   self.ddp)

        logger.info("Initiating the local training")
        client.initiate_local_training()

        logger.info("Local training starts ... ")
        client.train()

        logger.info("\nTerminating the local training")
        model = client.terminate_local_training(self.local_comm_round_idx)

        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        return buffer.getvalue()

    """
    evaluate() should:
    1. Take in the model weights as bytes and load them into your model
    3. If parameters passed are None, initialize them to match your untrained model's parameters (i.e. clean slate)
    4. If needed pre-process the dataset which is passed as a list of rows parsed as dicts
    5. Output the accuracy of the model parameters on the dataset as a float
    """

    def evaluate(self, parameters: bytes | None) -> float:
        local_train_dataset, local_eval_dataset = self.load_dataset(self.generate_and_tokenize_prompt, local_val_set_size)

        model = self.get_starting_model()
        if parameters is not None:
            set_peft_model_state_dict(model, torch.load(io.BytesIO(parameters)), "default")

        # TODO - implement evaluation
        accuracy = 0.0
        test_loss = 0.0
        test_total = 0.0

        logger.info(
            f"Tobe implemented model test, Acc: {accuracy}, Loss: {round(test_loss / test_total, 4)}"
        )

        return accuracy

    """
    aggregate() should take in a list of model weights (bytes),
    aggregate them using avg and output the aggregated parameters as bytes.
    """

    def aggregate(self, parameters_list: list[bytes]) -> bytes:
        parameters_list = [
            torch.load(io.BytesIO(parameters)) for parameters in parameters_list
        ]
        averaged_params_template = parameters_list[0]
        for k in averaged_params_template.keys():
            temp_w = []
            for local_w in parameters_list:
                temp_w.append(local_w[k])
            averaged_params_template[k] = sum(temp_w) / len(temp_w)

        # Save the averaged parameters to the file
        torch.save(averaged_params_template,
                   os.path.join(self.output_dir, str(self.local_comm_round_idx), "adapter_model.bin"))
        self.lora_config.save_pretrained(self.output_dir)

        # Create a buffer
        buffer = io.BytesIO()

        # Save state dict to the buffer
        torch.save(averaged_params_template, buffer)

        # Get the byte representation
        aggregated_parameters = buffer.getvalue()

        return aggregated_parameters


if __name__ == "__main__":

    client_id = 1

    """
    Hyper parameters
    """
    global_model = './model/vicuna-7b-all-v1.1'
    data_path = './data'
    output_dir = './vicuna-lora-shepherd-7b/'
    # FL hyperparamas
    client_selection_frac = 0.1
    num_communication_rounds = 50
    num_clients = 10
    # Local training hyperparams
    local_batch_size = 64  # 64,
    local_micro_batch_size = 8
    local_num_epochs = 10
    local_learning_rate = 3e-4
    local_val_set_size = 0
    local_save_steps = 3
    cutoff_len = 1024
    # LoRA hyperparams
    lora_r = 16
    lora_alpha = 16
    lora_dropout = 0.05
    lora_target_modules = [
        "q_proj",
    ]
    # lora_target_modules = [
    #     "q_proj","k_proj","v_proj","o_proj",
    # ]
    # llm hyperparams
    train_on_inputs = True
    group_by_length = False
    resume_from_checkpoint = None  # either training checkpoint or final adapter
    prompt_template_name = "alpaca"

    flock_model = FlockModel(
        global_model=global_model,
        data_path=data_path,
        output_dir=output_dir,
        # FL hyperparamas
        client_selection_frac=client_selection_frac,
        num_communication_rounds=num_communication_rounds,
        num_clients=num_clients,
        # Local training hyperparams
        local_batch_size=local_batch_size,
        local_micro_batch_size=local_micro_batch_size,
        local_num_epochs=local_num_epochs,
        local_learning_rate=local_learning_rate,
        local_val_set_size=local_val_set_size,
        local_save_steps=local_save_steps,
        cutoff_len=cutoff_len,
        # LoRA hyperparams
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        # llm hyperparams
        train_on_inputs=train_on_inputs,
        group_by_length=group_by_length,
        resume_from_checkpoint=resume_from_checkpoint,
        prompt_template_name=prompt_template_name,

        client_id=client_id,
    )
    flock.register_train(flock_model.train)
    flock.register_evaluate(flock_model.evaluate)
    flock.register_aggregate(flock_model.aggregate)
    flock.run()