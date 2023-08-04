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
import transformers
from peft import get_peft_model_state_dict

class GeneralClient:
    def __init__(self,
                 client_id,
                 model,
                 local_train_dataset,
                 local_eval_dataset,
                 local_val_set_size,
                 output_dir,
                 model_eval_steps=40,
                 model_save_steps=40):
        self.client_id = client_id

        self.model = model
        self.output_dir = output_dir
        self.local_output_dir = os.path.join(self.output_dir, "local_trainer_saved", "local_output")
        self.local_train_dataset= local_train_dataset
        self.local_eval_dataset= local_eval_dataset
        self.local_val_set_size=local_val_set_size

        self.model_eval_steps = model_eval_steps
        self.model_save_steps = model_save_steps

    def build_local_trainer(self,
                            tokenizer,
                            local_micro_batch_size,
                            gradient_accumulation_steps,
                            local_num_epochs,
                            local_learning_rate,
                            group_by_length,
                            ddp):

        self.train_args = transformers.TrainingArguments(
            per_device_train_batch_size=local_micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=0,
            num_train_epochs=local_num_epochs,
            learning_rate=local_learning_rate,
            fp16=True,
            logging_steps=1,
            optim="adamw_torch",
            evaluation_strategy="steps" if self.local_val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=self.model_eval_steps if self.local_val_set_size > 0 else None,
            save_steps=self.model_save_steps,
            output_dir=self.local_output_dir,
            save_total_limit=1,
            load_best_model_at_end=True if self.local_val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            dataloader_drop_last=False
        )
        self.local_trainer = transformers.Trainer(model=self.model,
                                                  train_dataset=self.local_train_dataset,
                                                  eval_dataset=self.local_eval_dataset,
                                                  args=self.train_args,
                                                  data_collator=transformers.DataCollatorForSeq2Seq(
                                                      tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
                                                  ),
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
        single_output_dir = os.path.join(self.output_dir, str(local_comm_round_idx), "local_output_{}".format(self.client_id))
        os.makedirs(single_output_dir, exist_ok=True)
        torch.save(new_adapter_weight, single_output_dir + "/pytorch_local_model_lora.bin")

        return self.model
