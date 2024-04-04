from prompters.vicuna_prompter import VicunaPrompter
from typing import Union

class MixtralPrompter(VicunaPrompter):

    def __init__(self, template_name: str = "", verbose: bool = False):
        super().__init__(template_name, verbose)

    def generate_prompt(
            self,
            instruction: str,
            input: Union[None, str] = None,
            label: Union[None, str] = None,
    ) -> str:
        full_prompt = f"""<s>[INST]{instruction}
            {f"Here is some context: {input}" if len(input) > 0 else None}
             [/INST] {label}
            </s>"""
        return full_prompt