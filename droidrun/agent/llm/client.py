from typing import List, Any, Optional, Union, Tuple
import base64
from droidrun.agent.llm.provider import Provider


class LLMClient:
    def __init__(self, provider: Provider, system_prompt: Optional[str] = None):
        self.provider = provider
        self.messages = []
        if system_prompt:
            self.add_system_prompt(system_prompt)
        else:
            self.add_system_prompt("You are a helpful assistant.")

    def calculate_tokens(
        self,
        messages: Optional[List[Any]] = None,
        num_image_token: Optional[int] = None,
    ) -> Tuple[int, int]:
        if messages is None:
            messages = self.messages
        return self.provider.calculate_tokens(messages, num_image_token)

    def encode_image(self, image_content: Union[bytes, str]):
        # if image_content is a path to an image file, check type of the image_content to verify
        if isinstance(image_content, str):
            with open(image_content, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        else:
            return base64.b64encode(image_content).decode("utf-8")

    def reset(self):
        self.messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            }
        ]

    def add_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt
        if len(self.messages) > 0:
            self.messages[0] = {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            }
        else:
            self.messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                }
            )

    def remove_message_at(self, index: int):
        """Remove a message at a given index"""
        if index < len(self.messages):
            self.messages.pop(index)

    def replace_message_at(
        self,
        index: int,
        text_content: str,
        image_content: Optional[Union[str, List[str], bytes, List[bytes]]] = None,
        image_detail: str = "high",
    ):
        """Replace a message at a given index"""
        if index < len(self.messages):
            self.messages[index] = {
                "role": self.messages[index]["role"],
                "content": [{"type": "text", "text": text_content}],
            }
            if image_content:
                base64_image = self.encode_image(image_content)
                self.messages[index]["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": image_detail,
                        },
                    }
                )

    def infer_role(self, role: Optional[str] = None) -> str:
        # infer role from previous message
        if role != "user":
            if self.messages[-1]["role"] == "system":
                role = "user"
            elif self.messages[-1]["role"] == "user":
                role = "assistant"
            elif self.messages[-1]["role"] == "assistant":
                role = "user"

        return role

    def add_message(
        self,
        text_content: str,
        image_content: Optional[Union[str, List[str], bytes, List[bytes]]] = None,
        role: Optional[str] = None,
        image_detail: str = "high",
        put_text_last: bool = False,
    ):
        """Add a new message to the list of messages"""
        role = self.infer_role(role)
        self.messages.append(
            self.provider.format_message(
                role, text_content, image_content, image_detail, put_text_last
            )
        )

    async def get_response(
        self,
        user_message: Optional[any] = None,
        messages: Optional[List[Any]] = None,
        temperature: Optional[float] = 0.0,
        max_new_tokens: Optional[int] = None,
        **kwargs,
    ):
        """Generate the next response based on previous messages"""
        if messages is None:
            messages = self.messages

        if user_message:
            messages.append(
                {"role": "user", "content": [{"type": "text", "text": user_message}]}
            )

        return await self.provider.generate(
            messages,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
