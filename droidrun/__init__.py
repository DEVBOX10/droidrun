"""
DroidRun - A framework for controlling Android devices through LLM agents.
"""

__version__ = "0.2.0"

# Import main classes for easier access
from droidrun.agent.droidrun import DroidRunAgent as Agent

# Make main components available at package level
__all__ = [
    "Agent",
] 