📁 **Prompt Templates Directory**

Welcome to the central repository for template styles used in fine-tuning LoRA models.

## 📌 **Structure & Formatting**

Each template is housed in a JSON file and includes the following attributes:

- 📜 `prompt_input`: This is used when an input is provided. It can accommodate `{instruction}` and `{input}` placeholders.
- 🚫 `prompt_no_input`: This is employed when there's no available input and includes the `{instruction}` placeholder.
- 📝 `description`: Provides a concise explanation and details potential use cases for the template.
- ✂️ `response_split`: Identifies the delimiter for separating the actual model response from the output.

> 🚨 **Note**: The `{response}` placeholder is absent, as the response is always the last element of the template, appended to the existing content.

## 📄 **Example Template**

Unless otherwise specified, the default template is `alpaca.json`:

```json
{
    "description": "This template is used by Alpaca-LoRA.",
    "prompt_input": "The content below presents a task description and relevant context. Please draft a response that meets the requirement.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
    "prompt_no_input": "The instruction below defines a specific task. Create a response that appropriately fulfills the requirement.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
    "response_split": "### Response:"    
}
```

📚 Available Templates
1. **alpaca** 🦙:
   The universal template used in most LoRA fine-tuning tasks.
2. **alpaca_legacy** ⌛:
   An original version from the alpaca repository, notable for lacking a newline (\n) after the response section. Retained for comparative studies and experiments.
3. **alpaca_short** 📏:
   A brief version of the alpaca template, offering similar effectiveness while conserving tokens. Models developed with the main template can also be prompted using this shortened variant. Further experimentation and contributions are welcome.
4. **vigogne** 🦙:
   This is the French version of the alpaca template. Used for training the "Vigogne" LoRA, it is suitable for inquiries or additional fine-tuning processes.

## 📖 **References**

- [Shepherd](https://github.com/JayZhang42/FederatedGPT-Shepherd)