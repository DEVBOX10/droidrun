from typing import List, Any, Tuple, Optional, Union
from droidrun.agent.llm.client import Provider
from openai import OpenAI
import asyncio
import tiktoken

NUM_IMAGE_TOKEN = 1105  # Value set of screen of size 1920x1080 for openai vision


class OpenAIProvider(Provider):
    def __init__(self, model: str, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def format_message(
        self,
        role: str,
        text_content: str,
        image_content: Optional[Union[str, List[str], bytes, List[bytes]]] = None,
        image_detail: str = "high",
        put_text_last: bool = False,
    ):
        message = {
            "role": role,
            "content": [{"type": "text", "text": text_content}],
        }

        if image_content:
            # Check if image_content is a list or a single image
            if isinstance(image_content, list):
                # If image_content is a list of images, loop through each image
                for image in image_content:
                    base64_image = self.encode_image(image)
                    message["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": image_detail,
                            },
                        }
                    )
            else:
                # If image_content is a single image, handle it directly
                base64_image = self.encode_image(image_content)
                message["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": image_detail,
                        },
                    }
                )

        # Rotate text to be the last message if desired
        if put_text_last:
            text_content = message["content"].pop(0)
            message["content"].append(text_content)

        return message

    async def generate(
        self,
        messages: List[Any],
        temperature: float = 0.0,
        max_new_tokens: int = 1000,
        **kwargs,
    ) -> str:
        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens if max_new_tokens else 4096,
            **kwargs,
        )

        return response.choices[0].message.content

    def calculate_tokens(
        self, messages: List[Any], num_image_token: Optional[int] = None
    ) -> Tuple[int, int]:
        if not num_image_token:
            num_image_token = NUM_IMAGE_TOKEN

        num_input_images = 0
        output_message = messages[-1]

        input_message = messages[:-1]

        input_string = """"""
        for message in input_message:
            input_string += message["content"][0]["text"] + "\n"
            if len(message["content"]) > 1:
                num_input_images += 1

        input_text_tokens = get_input_token_length(input_string)

        input_image_tokens = num_image_token * num_input_images

        output_tokens = get_input_token_length(output_message["content"][0]["text"])

        return (input_text_tokens + input_image_tokens), output_tokens

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        # Set Cost based on GPT-4o
        return input_tokens * (0.0050 / 1000) + output_tokens * (0.0150 / 1000)


def get_input_token_length(input_string):
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(input_string)
    return len(tokens)
