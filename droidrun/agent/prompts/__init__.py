from string import Template
import os
import logging

logger = logging.getLogger("prompts")
current_dir = os.path.dirname(__file__)


class Prompt:
    def __init__(self, prompt_name: str):
        self.template = self._load_template(prompt_name)

    def _load_template(self, prompt_name: str) -> Template:
        file_path = os.path.join(current_dir, f"{prompt_name}.md")
        logger.info(f"Loading prompt template from {file_path}")
        with open(file_path, "r") as f:
            return Template(f.read())

    def build(self, **kwargs) -> str:
        return self.template.substitute(**kwargs)
