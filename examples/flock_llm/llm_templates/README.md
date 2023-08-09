# 📁 **Prompt Templates Directory**

Welcome to the central repository for template styles utilized in fine-tuning LoRA models.

## 📌 **Structure & Formatting**

Each template is encapsulated in a JSON file and contains the following attributes:

- 📜 `prompt_input`: Utilized when an input is provided. It accommodates `{instruction}` and `{input}` placeholders.
- 🚫 `prompt_no_input`: Deployed when no input is available. It encompasses the `{instruction}` placeholder.
- 📝 `description`: Offers a succinct explanation and potential use cases of the template.
- ✂️ `response_split`: Designates the delimiter for segregating the actual model response from the output.

> 🚨 **Note**: The `{response}` placeholder isn't present, as the response is consistently the trailing element of the template, and it simply appends to the existing content.

## 📄 **Example Template**

Unless specified otherwise, the default template is `alpaca.json`:

```json
{
    "description": "Template utilized by Alpaca-LoRA.",
    "prompt_input": "The subsequent content presents a task description combined with relevant context. Please draft a response that fulfills the requirement.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
    "prompt_no_input": "The instruction below outlines a specific task. Construct a response that aptly fulfills the requirement.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
    "response_split": "### Response:"    
}
```

📚 Available Templates
1. alpaca 🦙
The universal template utilized in the majority of LoRA fine-tuning endeavors.
2. alpaca_legacy ⌛
An earlier version from the original alpaca repository, distinguished by the absence of a newline (\n) succeeding the response section. Retained for comparative studies and experimental endeavors.
3. alpaca_short 📏
A concise version of the alpaca template, offering comparable efficacy whilst conserving tokens. It's observed that models crafted with the principal template can also be prompted using this shortened variant. Contributions via further experimentation are encouraged.
4. vigogne 🦙
This is the alpaca template translated into French. Employed for training the "Vigogne" LoRA, it should be used for making inquiries or for additional fine-tuning processes.

## 📖 **References**

- [Shepherd](https://github.com/JayZhang42/FederatedGPT-Shepherd)