<div align="center">

# Configuration YAML file parameters explanation

</div>

|          Category           | Version                            | Default                       |   Type    | Options                                                     | Description                               |
|:---------------------------:|:-----------------------------------|:------------------------------|:---------:|:------------------------------------------------------------|:------------------------------------------|
|         common_args         | project_name                       | "FLockLLM_finetune"           |    str    | -                                                           | Overall project name                      |
|                             | random_seed                        | 1993                          |    int    | -                                                           | Random Seed                               |
|                             | mode                               | "experiment"                  |    str    | "experiment", "deployment"                                  | -                                         |
|          data_args          | data_path                            | "dataset.json"                |    str    | "dataset.json"                                              | -                                         |
|         model_args          | foundation_model                              | "gemma-2b"                    |    str    | "gemma-2b", "mixtral7b", "vicuna-7b-v1.5"                   | -                                         |
|                             | foundation_model_pre_trained_weights_path               | model/gemma-2b                |    str    | "model/gemma-2b", "model/Mistral-7B-v0.1", "model/vicuna-7b-v1.5" | -                                         |
|                             | lora_dropout                         | 0.05                          |   float   | 0.05                                                        | -                                         |
|                             | lora_target_modules                         | []                            | list(str) | "q_proj","k_proj","v_proj","o_proj"                         | [] (let system auto search)               |
|         train_args          | federated_optimizer                | "fedavg"                      |    str    | "fedavg"                                                    | Federated learning optimization algorithm |
|                             | proposer_train_batch_size                 | 32                            |    int    | -                                                           | -                                         |
|                             | proposer_train_micro_batch_size | 8                             |    int    | -                                                           | -                                         |
|                             | proposer_num_epochs                   | 1                             |    int    | -                                                           | -                                         |
|                             | proposer_learning_rate                    | 0.0003                          |   float   | -                                                           | -                                         |
|                             | proposer_val_set_size                   | 0                             |    int    | -                                                           | -                                         |
|                             | cutoff_len                | 512                           |    int    | -                                                           | -                                         |
|                             | proposer_train_group_by_length                    | false                         |   bool    | true/false                                                  | -                                         |
|                             | proposer_train_optimizer     | "paged_adamw_8bit"            |    str    | true/false                                                  | -                                         |
|                             | proposer_train_lr_scheduler_type     | "constant"                    |   bool    | true/false                                                  | -                                         |
|                             | proposer_train_warmup_steps                   | 1                             |    int    | -                                                           | -                                         |
|                             | proposer_train_weight_decay    | 0.05                          |   float   | -                                                           | -                                         |
|                             | proposer_train_block_size      | 8                             |    int    | -                                                           | -                                         |
|       evaluation_args       | voter_val_set_size           | 5                             |    int    | -                                                           | -                                         |
|        tracking_args        | finetune_adapter_checkpoint_save_dir                       | "output/checkpoints/gemma-2b" |    str    | -                                                           | -                                         |
|                             | proposer_train_gradient_checkpointing                          | true                          |   bool    | true/false                                                  | -                                         |
|                             | proposer_train_logging_steps                       | 10                            |    int    | -                                                           | -                                         |
|                             | proposer_save_steps                      | 3                             |    int    | -                                                           | -                                         |
|                             | report_to                           | "wandb"                       |    str    | "wandb"                                                     | -                                         |
|                             | save_total_limit    | 3                             |    int    | -                                                           | -                                         |
|                             |                                    |                               |           |                                                             |                                           |
|                             |                                    |                               |           |                                                             |                                           |
|                             |                                    |                               |           |                                                             |                                           |
|                             |                                    |                               |           |                                                             |                                           |
|                             |                                    |                               |           |                                                             |                                           |
|                             |                                    |                               |           |                                                             |                                           |
|                             |                                    |                               |           |                                                             |                                           |
|                             |                                    |                               |           |                                                             |                                           |
|                             |                                    |                               |           |                                                             |                                           |

