"""

FLock LLM example code based on the FLock sdk

"""
import io
import os
import datasets as pypi_datasets
from datasets import load_dataset as pypi_load_dataset
from utils.helper import print_number_of_trainable_model_parameters
import torch

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, AutoConfig

import bitsandbytes as bnb
from trl import SFTTrainer

from loguru import logger

from utils.helper import test_mkdir
from flock_sdk import FlockModel
from client import GeneralClient
from prompters.prompter_hub import get_prompter

pypi_datasets.utils.logging.set_verbosity_error()

class FLockLLMFinetuneModel(FlockModel):
    def __init__(
            self,
            args,
    ):

        self.args = args

        # Model args
        self.model_name = args.foundation_model
        self.global_model_path = args.foundation_model_pre_trained_weights_path
        self.finetune_adapter = args.finetune_adapter

        if args.finetune_adapter.lower() == "lora":
            self.lora_r = 16
            self.lora_alpha = 16
        elif args.finetune_adapter.lower() == "qlora":
            self.lora_r = 4
            self.lora_alpha = 4
        else:
            raise ValueError(f"Adapter type {self.finetune_adapter} not recognized")
        self.lora_dropout = args.lora_dropout
        self.lora_target_modules = args.lora_target_modules


        # Train args
        self.local_batch_size = args.proposer_train_batch_size
        self.local_micro_batch_size = args.proposer_train_micro_batch_size
        self.local_num_epochs = args.proposer_num_epochs
        self.local_learning_rate = args.proposer_learning_rate
        self.local_val_set_size = args.proposer_val_set_size
        self.voter_val_set_size = args.voter_val_set_size
        self.local_save_steps = args.proposer_save_steps
        self.cutoff_len = args.cutoff_len
        self.group_by_length = args.proposer_train_group_by_length
        self.optim = args.proposer_train_optimizer
        self.lr_scheduler_type = args.proposer_train_lr_scheduler_type
        self.warmup_steps = args.proposer_train_warmup_steps
        self.weight_decay = args.proposer_train_weight_decay
        self.block_size = args.proposer_train_block_size


        # Tracking args
        self.output_dir = args.finetune_adapter_checkpoint_save_dir
        self.gradient_checkpointing = args.proposer_train_gradient_checkpointing
        self.logging_steps = args.proposer_train_logging_steps
        self.report_to = args.report_to
        self.save_total_limit = args.save_total_limit

        logger.debug(
            f"FLockLLM finetuning using LoRA with params:\n"
            f"global_model: {self.global_model_path}\n"
            f"output_dir: {self.output_dir}\n"
            f"local_batch_size: {self.local_batch_size}\n"
            f"local_micro_batch_size: {self.local_micro_batch_size}\n"
            f"local_num_epochs: {self.local_num_epochs}\n"
            f"local_learning_rate: {self.local_learning_rate}\n"
            f"local_val_set_size: {self.local_val_set_size}\n"
            f"local_save_steps: {self.local_save_steps}\n"
            f"cutoff_len: {self.cutoff_len}\n"
            f"lora_r: {self.lora_r}\n"
            f"lora_alpha: {self.lora_alpha}\n"
            f"lora_dropout: {self.lora_dropout}\n"
            f"lora_target_modules: {self.lora_target_modules}\n"
            f"group_by_length: {self.group_by_length}\n"
            f"gradient_checkpointing: {self.gradient_checkpointing or False}\n"
            # f"prompt template: {prompt_template_name}\n"
            f"logging_steps: {self.logging_steps}\n"
            f"optim: {self.optim}\n"
            f"lr_scheduler_type: {self.lr_scheduler_type}\n"
            f"warmup_steps: {self.warmup_steps}\n"
            f"weight_decay: {self.weight_decay}\n"
            f"report_to: {self.report_to}\n"
            f"save_total_limit: {self.save_total_limit}\n"
            f"block_size: {self.block_size}\n"
        )

        if torch.cuda.is_available():
            logger.debug("CUDA is available. Here are the device details:")
            # 获取CUDA设备数量
            num_devices = torch.cuda.device_count()
            logger.debug(f"Number of CUDA devices available: {num_devices}")

            # 遍历每个CUDA设备
            for i in range(num_devices):
                logger.debug(f"Device {i}: {torch.cuda.get_device_name(i)}")
                # 获取当前设备的详细信息
                device_properties = torch.cuda.get_device_properties(i)
                logger.debug(f"  Memory Allocation: {device_properties.total_memory / 1e9} GB")
                logger.debug(f"  CUDA Capability: {device_properties.major}.{device_properties.minor}")
        else:
            logger.warning("CUDA is not available.")

        self.local_comm_round_idx = 0

        """
            LLM Settings & Preparation
        """

        self.prompter = get_prompter()
        self.gradient_accumulation_steps = self.local_batch_size // self.local_micro_batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(self.global_model_path,
                                                       trust_remote_code=False,
                                                       use_fast=True)

        pad_token_id = 0
        self.tokenizer.pad_token_id = pad_token_id
        self.tokenizer.pad_token = self.tokenizer.convert_ids_to_tokens(pad_token_id)
        self.tokenizer.padding_side = "right"  # "left"

        """
            Device and DDP setting
        """
        # TODO temp solution
        if "gemma" in self.model_name:
            self.device_map = "cuda:0"
        else:
            self.device_map = "auto"

        world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.ddp = world_size != 1
        self.global_model = self.get_model()

        if self.ddp:
            self.device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
            self.gradient_accumulation_steps = self.gradient_accumulation_steps // world_size

        """
            Dataset loading
        """
        self.local_train_dataset, self.local_eval_dataset = self.init_dataset(self.args.data_path)

    def init_dataset(self, dataset_path: str):
        logger.info("\nPreparing the local training and validation dataset")

        local_data = pypi_load_dataset("json", data_files=dataset_path)

        if self.voter_val_set_size > 0:
            split_params = {
                "test_size": self.voter_val_set_size,
                "shuffle": True
            }
            if hasattr(self.args, 'random_seed'):
                split_params['seed'] = self.args.random_seed
            local_train_val = local_data["train"].train_test_split(**split_params)

            self.local_train_dataset = (
                local_train_val["train"].shuffle().map(self.generate_and_tokenize_prompt)
            )
            self.local_eval_dataset = (
                local_train_val["test"].shuffle().map(self.generate_and_tokenize_prompt)
            )
        else:
            self.local_train_dataset = local_data["train"].shuffle().map(self.generate_and_tokenize_prompt)
            self.local_eval_dataset = None

        return self.local_train_dataset, self.local_eval_dataset

    def tokenize(self, prompt, add_eos_token=True):
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.cutoff_len,
            padding="max_length",
            return_tensors=None,
        )

        if (
                result["input_ids"][-1] != self.tokenizer.eos_token_id
                and len(result["input_ids"]) < self.cutoff_len
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

        tokenized_full_prompt = {"text": full_prompt}

        return tokenized_full_prompt

    def get_model(self):
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        if self.finetune_adapter.lower() == "lora":
            model = AutoModelForCausalLM.from_pretrained(self.global_model_path,
                                                         load_in_8bit=True,
                                                         trust_remote_code=False,
                                                         device_map=self.device_map)
            model = prepare_model_for_int8_training(model)
        elif self.finetune_adapter.lower() == "qlora":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",  # 量化时使用nf4最优，也可以使用fp4
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,  # 二次量化
            )

            config = AutoConfig.from_pretrained(self.global_model_path)
            config.use_cache = False
            config.gradient_checkpointing = True

            model = AutoModelForCausalLM.from_pretrained(self.global_model_path,
                                                         config=config,
                                                         quantization_config=bnb_config,
                                                         trust_remote_code=False,
                                                         torch_dtype=torch_dtype,
                                                         device_map=self.device_map)
            model = prepare_model_for_kbit_training(model,
                                                    use_gradient_checkpointing=True)
        else:
            raise ValueError(f"Adapter type {self.finetune_adapter} not recognized")

        def find_all_linear_names(model, add_lm_head=True):
            cls = bnb.nn.Linear4bit
            lora_module_names = set()
            for name, module in model.named_modules():
                if isinstance(module, cls):
                    names = name.split('.')
                    lora_module_names.add(names[0] if len(names) == 1 else names[-1])

            if add_lm_head and not "lm_head" in lora_module_names:
                lora_module_names.add("lm_head")

            return list(lora_module_names)

        if len(self.lora_target_modules) == 0:
            self.lora_target_modules = find_all_linear_names(model)

        self.lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.lora_target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            # inference_mode=False
        )

        # Inject QLoRA into pre-trained model

        model = get_peft_model(model, self.lora_config)

        if self.ddp and torch.cuda.device_count() > 1:
            model.is_parallelizable = True
            model.model_parallel = True

        print_number_of_trainable_model_parameters(model)

        return model

    def train(self, parameters) -> bytes:
        self.local_comm_round_idx += 1

        # Load model template with pre-trained weights
        model = self.global_model
        if parameters is not None:
            logger.debug("Loading latest global adapter model parameters to local model...")
            set_peft_model_state_dict(model, torch.load(io.BytesIO(parameters)), "default")

        model.train()

        client = GeneralClient(model=model, local_train_dataset=self.local_train_dataset, local_eval_dataset=None,
                               local_val_set_size=self.local_val_set_size, output_dir=self.output_dir)
        client.build_local_trainer(model_name=self.model_name,
                                   tokenizer=self.tokenizer,
                                   local_micro_batch_size=self.local_micro_batch_size,
                                   gradient_accumulation_steps=self.gradient_accumulation_steps,
                                   local_num_epochs=self.local_num_epochs,
                                   local_learning_rate=self.local_learning_rate,
                                   group_by_length=self.group_by_length,
                                   logging_steps=self.logging_steps,
                                   optim=self.optim,
                                   lr_scheduler_type=self.lr_scheduler_type,
                                   warmup_steps=self.warmup_steps,
                                   weight_decay=self.weight_decay,
                                   report_to=self.report_to,
                                   save_total_limit=self.save_total_limit,
                                   block_size=self.block_size,
                                   gradient_checkpointing=self.gradient_checkpointing,
                                   ddp=self.ddp,
                                   )

        logger.info("Initiating the local training...")
        client.initiate_local_training()

        logger.info("Local training starts...")
        client.train()

        logger.info("Terminating the local training...")
        model = client.terminate_local_training(self.local_comm_round_idx)

        logger.info("Wrapping up the local model parameters and sending to voters...")
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        return buffer.getvalue()

    def evaluate(self, parameters: bytes) -> float:
        model = self.global_model
        if parameters is not None:
            logger.debug("\nLoading latest global adapter model parameters to local model...")
            set_peft_model_state_dict(model, torch.load(io.BytesIO(parameters)), "default")

        eval_args = TrainingArguments(
            do_train=False,
            do_eval=True,
            output_dir=self.output_dir,
        )

        trainer = SFTTrainer(
            model=model,
            args=eval_args,
            eval_dataset=self.local_eval_dataset,
            dataset_text_field="text",
            max_seq_length=self.block_size,
            tokenizer=self.tokenizer,
            data_collator=None,
            packing=None
        )

        logger.info(
            f"Global adapter model evaluation start..."
        )

        eval_result = trainer.evaluate()

        logger.info(
            f"Global adapter model loss: {round(eval_result['eval_loss'], 6)}"
        )

        # Using miners for temp
        return eval_result['eval_loss']

    def aggregate(self, parameters_list: list[bytes]) -> bytes:
        # Handle DDP alignment problem: relocate the model weights to unified device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        parameters_list = [
            torch.load(io.BytesIO(parameters), map_location=device) for parameters in parameters_list
        ]

        logger.info("Aggregating the all local model parameters...")
        if self.args.federated_optimizer.lower() == "fedavg":
            averaged_params_template = parameters_list[0]
            for k in averaged_params_template.keys():
                temp_w = []
                for local_w in parameters_list:
                    temp_w.append(local_w[k])
                averaged_params_template[k] = sum(temp_w) / torch.tensor(len(temp_w)).to(device)
        else:
            raise NotImplementedError(f"The federated optimizer ({self.args.federated_optimizer}) is not supported.")

        # Check output dir
        target_path = os.path.join(self.output_dir, str(self.local_comm_round_idx))
        test_mkdir(target_path)

        # Save the averaged parameters to the file
        global_model_output_path = os.path.join(target_path, "pytorch_local_model_lora.bin")
        logger.info(f"Saving the global adapter model parameters to {global_model_output_path}...")
        torch.save(averaged_params_template,
                   global_model_output_path)
        self.lora_config.save_pretrained(self.output_dir)

        logger.info("Wrapping up the global adapter model parameters and sending to all Proposers...")
        # Create a buffer
        buffer = io.BytesIO()
        # Save state dict to the buffer
        torch.save(averaged_params_template, buffer)
        # Get the byte representation
        aggregated_parameters = buffer.getvalue()

        return aggregated_parameters
