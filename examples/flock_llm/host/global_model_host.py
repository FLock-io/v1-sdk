"""

Designed for output global model hosting.

Reference:
    1. Shepherd: A Lightweight GitHub Platform Supporting Federated Instruction Tuning
        - https://github.com/JayZhang42/FederatedGPT-Shepherd
        - Jianyi Zhang and Martin Kuo and Ruiyi Zhang and Guoyin Wang and Saeed Vahidian and Yiran Chen
"""

import os

import fire
import torch
import transformers
from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM
import gradio as gr
from peft import (
    PeftModel,
    LoraConfig,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from .callbacks import Iteratorize, Stream
from prompters.prompter_hub import get_prompter

# Check if we have a GPU available
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

def main(
    load_8bit: bool = False,
    base_model_path: str = "",
    lora_weights_path: str = "",
    lora_config_path: str= "", # provide only the file path, excluding the file name 'adapter_config.json'
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    server_name: str = "127.0.0.1",
    share_gradio: bool = False,
):
    base_model_path = base_model_path or os.environ.get("BASE_MODEL", "")
    assert (
        base_model_path
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = get_prompter()
    tokenizer = AutoTokenizer.from_pretrained(base_model_path,
                                                       trust_remote_code=False,
                                                       use_fast=True)
    if not lora_weights_path.endswith(".bin"):
        if device == "cuda":

            model = AutoModelForCausalLM.from_pretrained(base_model_path,
                                                         load_in_8bit=load_8bit,
                                                         trust_remote_code=False,
                                                         device_map="auto")
            model = PeftModel.from_pretrained(
                model,
                lora_weights_path,
                torch_dtype=torch.float16,
            )
        elif device == "mps":
            model = AutoModelForCausalLM.from_pretrained(base_model_path,
                                                         load_in_8bit=load_8bit,
                                                         trust_remote_code=False,
                                                         torch_dtype=torch.float16,
                                                         device_map={"": device})
            model = PeftModel.from_pretrained(
                model,
                lora_weights_path,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(base_model_path,
                                                         load_in_8bit=load_8bit,
                                                         trust_remote_code=False,
                                                         low_cpu_mem_usage=True,
                                                         device_map={"": device})
            model = PeftModel.from_pretrained(
                model,
                lora_weights_path,
                device_map={"": device},
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(base_model_path,
                                                     load_in_8bit=load_8bit,
                                                     trust_remote_code=False,
                                                     torch_dtype=torch.float16,
                                                     device_map="auto")
        model = prepare_model_for_int8_training(model)
        lora_config = LoraConfig.from_pretrained(lora_config_path)

        lora_weights = torch.load(lora_weights_path, map_location=device)
        model = PeftModel(model, lora_config)
        set_peft_model_state_dict(model,lora_weights,"default")
        del lora_weights

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()

    def evaluate(
        data_point,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        stream_output=True,
        **kwargs,
    ):
        """
        Generate model human evaluation setting and WebUI.
        """
        # prompt = prompter.generate_prompt(instruction, input)

        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["context"],
            data_point["response"],
        )

        inputs = tokenizer(full_prompt, return_tensors="pt")

        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }

        if stream_output:
            # Stream the reply 1 token at a time.
            # This is based on the trick of using 'stopping_criteria' to create an iterator,
            # from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/text_generation.py#L216-L243.

            def generate_with_callback(callback=None, **kwargs):
                kwargs.setdefault(
                    "stopping_criteria", transformers.StoppingCriteriaList()
                )
                kwargs["stopping_criteria"].append(
                    Stream(callback_func=callback)
                )
                with torch.no_grad():
                    model.generate(**kwargs)

            def generate_with_streaming(**kwargs):
                return Iteratorize(
                    generate_with_callback, kwargs, callback=None
                )

            with generate_with_streaming(**generate_params) as generator:
                for output in generator:
                    # new_tokens = len(output) - len(input_ids[0])
                    decoded_output = tokenizer.decode(output)

                    if output[-1] in [tokenizer.eos_token_id]:
                        break

                    yield prompter.get_response(decoded_output)
            return  # early return for stream_output

        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        yield prompter.get_response(output)

    FLockLLM_UI=gr.Interface(
        fn=evaluate,
        inputs=[
            gr.components.Textbox(
                lines=2,
                label="Instruction",
                placeholder="Tell me about alpacas.",
            ),
            gr.components.Textbox(lines=2, label="Input", placeholder="none"),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.1, label="Temperature"
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.75, label="Top p"
            ),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=40, label="Top k"
            ),
            gr.components.Slider(
                minimum=1, maximum=4, step=1, value=4, label="Beams"
            ),
            gr.components.Slider(
                minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
            ),
            gr.components.Checkbox(label="Stream output"),
        ],
        outputs=[
            gr.inputs.Textbox(
                lines=5,
                label="Output",
            )
        ],
        title="FLockLLM",
        description="FLockLLM is a Large Language Model that has been fine-tuned in a web3-based, decentralized manner.",
    ).queue()

    # FLockLLM_UI.launch(auth=("admin", "pass1234"), server_name='0.0.0.0', server_port=8080, show_error=True,share=True)
    FLockLLM_UI.launch(auth=("admin", "admin"), server_name='0.0.0.0', show_error=True, share=True)


if __name__ == "__main__":
    fire.Fire(main)