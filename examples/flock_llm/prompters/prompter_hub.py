from prompters.mixtral_prompter import MixtralPrompter
from prompters.vicuna_prompter import VicunaPrompter


# def get_prompter(prompter_type: str, template_name: str = "", verbose: bool = False):
def get_prompter(template_name: str = "", verbose: bool = False):
    # if "vicuna" in prompter_type.lower():
    #     return VicunaPrompter(template_name, verbose)
    # elif "mixtral" in prompter_type.lower() or "gemma" in prompter_type.lower():
    #     return MixtralPrompter(template_name, verbose)
    # else:
    #     raise ValueError(f"Unknown prompter type: {prompter_type}")
    return MixtralPrompter(template_name, verbose)