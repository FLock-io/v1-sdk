"""

Federated Learning Client

Reference:
    1. Shepherd: A Lightweight GitHub Platform Supporting Federated Instruction Tuning
        - https://github.com/JayZhang42/FederatedGPT-Shepherd
        - Jianyi Zhang and Martin Kuo and Ruiyi Zhang and Guoyin Wang and Saeed Vahidian and Yiran Chen
"""

import os
from collections import OrderedDict

import torch
from peft import get_peft_model_state_dict
from trl import SFTTrainer
from transformers import TrainingArguments
import transformers

class GeneralClient:
    def __init__(self,
                 model,
                 local_train_dataset,
                 local_eval_dataset,
                 local_val_set_size,
                 output_dir,
                 model_eval_steps=40,
                 model_save_steps=40):
        self.model = model
        self.output_dir = output_dir
        self.local_output_dir = os.path.join(self.output_dir, "local_trainer_saved", "local_output")
        self.local_train_dataset= local_train_dataset
        self.local_eval_dataset= local_eval_dataset
        self.local_val_set_size=local_val_set_size

        self.model_eval_steps = model_eval_steps
        self.model_save_steps = model_save_steps

    def build_local_trainer(self,
                            model_name,
                            tokenizer,
                            local_micro_batch_size,
                            gradient_accumulation_steps,
                            local_num_epochs,
                            local_learning_rate,
                            logging_steps,
                            optim,
                            lr_scheduler_type,
                            warmup_steps,
                            weight_decay,
                            report_to,
                            save_total_limit,
                            block_size,
                            gradient_checkpointing,
                            group_by_length,
                            ddp):

        self.train_args = TrainingArguments(
            do_train=True,
            do_eval=True,
            output_dir=self.local_output_dir,
            dataloader_drop_last=False,  # Ori: True
            evaluation_strategy="steps" if self.local_val_set_size > 0 else "no",
            save_strategy="steps",
            logging_strategy="steps",
            num_train_epochs=local_num_epochs,
            eval_steps=self.model_eval_steps if self.local_val_set_size > 0 else None,
            save_steps=self.model_save_steps,
            logging_steps=logging_steps,
            per_device_train_batch_size=local_micro_batch_size,
            per_device_eval_batch_size=local_micro_batch_size * 2,
            optim=optim,
            learning_rate=local_learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            warmup_steps=warmup_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,
            weight_decay=weight_decay,
            report_to=report_to,
            load_best_model_at_end=True if self.local_val_set_size > 0 else False,
            save_total_limit=save_total_limit,
            bf16=True if torch.cuda.is_bf16_supported() else False,
            fp16=False if torch.cuda.is_bf16_supported() else True,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
        )

        self.local_trainer = SFTTrainer(
            model=self.model,
            args=self.train_args,
            train_dataset=self.local_train_dataset,
            eval_dataset=self.local_eval_dataset,
            dataset_text_field="text",
            max_seq_length=block_size,
            tokenizer=tokenizer,
            data_collator=None,
            packing=None,
        )

    def initiate_local_training(self):
        self.model.config.use_cache = False
        self.params_dict_new = OrderedDict((name, param.detach()) for name, param in self.model.named_parameters() if
                                           "default" in name)
        self.model.state_dict = (
            lambda instance, *_, **__: get_peft_model_state_dict(
                instance, self.params_dict_new, "default"
            )
        ).__get__(self.model, type(self.model))

    def train(self):
        self.local_trainer.train()

    def terminate_local_training(self, local_comm_round_idx):

        new_adapter_weight = self.model.state_dict()
        single_output_dir = os.path.join(self.output_dir, str(local_comm_round_idx), "local_output")
        os.makedirs(single_output_dir, exist_ok=True)
        torch.save(new_adapter_weight, single_output_dir + "/pytorch_local_model_lora.bin")

        return self.model
