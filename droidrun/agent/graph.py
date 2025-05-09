from pydantic import BaseModel, ValidationError
from typing import List
import json
import re


class State(BaseModel):
    name: str
    info: str

class Edge(BaseModel):
    source: str
    target: str
    event: str

class Graph(BaseModel):
    states: List[State]
    edges: List[Edge]


def parse_graph(text: str) -> Graph:
    pattern = r"<json>(.*?)</json>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            json_data = json.loads(json_str)
            return Graph(**json_data["graph"])
        except json.JSONDecodeError:
            print("Error: Invalid JSON")
            return None
        except KeyError:
            print("Error: 'graph' key not found in JSON")
            return None
        except ValidationError as e:
            print(f"Error: Invalid data structure - {e}")
            return None
    else:
        print("Error: JSON not found")
        return None
