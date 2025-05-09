from abc import ABC, abstractmethod
from typing import List, Any, Optional, Union, Tuple


class Provider(ABC):
    @abstractmethod
    async def generate(
        self,
        messages: List[Any],
        temperature: float = 0.0,
        max_new_tokens: int = 1000,
        **kwargs,
    ) -> str:
        pass

    @abstractmethod
    def format_message(
        self,
        role: str,
        text_content: str,
        image_content: Optional[Union[bytes, List[bytes]]] = None,
        image_detail: str = "high",
        put_text_last: bool = False,
    ) -> dict:
        pass

    @abstractmethod
    def calculate_tokens(
        self, messages: List[Any], num_image_token: Optional[int] = None
    ) -> Tuple[int, int]:
        pass

    @abstractmethod
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        pass
