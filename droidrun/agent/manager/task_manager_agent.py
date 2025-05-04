"""
Task Manager Agent - Delegates tasks to specialized agents.

This module implements a manager agent that routes tasks to the most appropriate
specialized agent based on the task's nature using LLM reasoning.
"""

import logging
from typing import Optional, Dict, Any, List
from ..base_llm_reasoner import BaseLLMReasoner

logger = logging.getLogger("droidrun")

DEFAULT_MANAGER_SYSTEM_PROMPT = '''You are an expert Task Manager that specializes in delegating tasks to the most appropriate specialized agent. Your job is to analyze a task and determine which expert agent would be best suited to handle it.

Available Expert Agents:

1. App Starter Agent
   - Specializes in launching Android applications
   - Excellent at determining correct package names
   - Best for tasks involving starting or launching apps
   - Example tasks: "open settings app", "launch camera", "start google maps"

2. ReAct Agent
   - Specializes in UI navigation and interaction
   - Can handle complex UI workflows
   - Best for tasks involving:
     * Navigating menus and screens
     * Filling forms
     * Interacting with UI elements
     * Reading and verifying UI state
   - Example tasks: "navigate to wifi settings", "enter password in the field", "toggle airplane mode"

Analyze the task and respond with a JSON object containing:
{
    "selected_agent": "app_starter" or "react",
    "confidence": float between 0 and 1,
    "reasoning": "Brief explanation of why this agent was selected"
}
'''

class TaskManagerAgent:
    """Manager agent that delegates tasks to specialized agents."""
    
    def __init__(
        self,
        llm: BaseLLMReasoner,
        react_agent,
        app_starter_agent,
        device_serial: Optional[str] = None,
        system_prompt: Optional[str] = None
    ):
        """Initialize the task manager agent.
        
        Args:
            llm: LLM reasoner for agent selection
            react_agent: ReAct agent for UI navigation tasks
            app_starter_agent: AppStarter agent for app launching tasks
            device_serial: Optional device serial number
            system_prompt: Optional custom system prompt
        """
        self.llm = llm
        self.react_agent = react_agent
        self.app_starter_agent = app_starter_agent
        self.device_serial = device_serial
        self.system_prompt = system_prompt or DEFAULT_MANAGER_SYSTEM_PROMPT
        
    async def execute_task(self, task: str) -> Dict[str, Any]:
        """Execute a task by delegating to the appropriate agent.
        
        Args:
            task: The task description from the planner
            
        Returns:
            Dictionary containing:
            - success: Whether the task was successful
            - steps: List of steps taken (if using ReAct agent)
            - action_count: Number of actions taken (if using ReAct agent)
            - error: Error message if task failed
            - agent_used: Which agent was selected for the task
            - confidence: Confidence in the agent selection
        """
        try:
            # Determine the best agent for this task using LLM
            agent_selection = await self._select_agent(task)
            selected_agent = agent_selection.get("selected_agent")
            confidence = agent_selection.get("confidence", 0)
            reasoning = agent_selection.get("reasoning", "")
            
            logger.info(f"Selected agent: {selected_agent} (confidence: {confidence})")
            logger.info(f"Reasoning: {reasoning}")
            
            if selected_agent == "app_starter" and confidence >= 0.7:
                logger.info("Delegating to AppStarter agent")
                result = await self.app_starter_agent.start_app(task)
                
                return {
                    "success": result.get("success", False),
                    "error": None if result.get("success", False) else "Failed to start app",
                    "agent_used": "app_starter",
                    "confidence": confidence
                }
            
            # Default to ReAct agent for all other cases
            logger.info("Delegating to ReAct agent")
            steps, action_count = await self.react_agent.run(task)
            
            # Check if task was successful
            task_success = False
            for step in reversed(steps):
                if step.step_type.value == "observation":
                    if "goal achieved" in step.content.lower():
                        task_success = True
                        break
            
            return {
                "success": task_success,
                "steps": steps,
                "action_count": action_count,
                "error": None if task_success else "Task execution failed",
                "agent_used": "react",
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Error executing task: {e}")
            return {
                "success": False,
                "error": str(e),
                "agent_used": None,
                "confidence": 0
            }
    
    async def _select_agent(self, task: str) -> Dict[str, Any]:
        """Use LLM to select the best agent for the task.
        
        Args:
            task: Task description
            
        Returns:
            Dictionary containing selected agent info
        """
        try:
            # Create the user prompt
            user_prompt = f"""
            Task: "{task}"
            
            Analyze this task and determine which specialized agent would be best suited to handle it.
            Consider the capabilities of each available agent and the nature of the task.
            Respond with a JSON object containing your selection and reasoning.
            """
            
            # Get LLM response
            response = await self.llm.generate_response(
                system_prompt=self.system_prompt,
                user_prompt=user_prompt
            )
            
            # Parse JSON response
            import json
            try:
                result = json.loads(response)
                return {
                    "selected_agent": result.get("selected_agent", "react"),  # Default to react if not specified
                    "confidence": float(result.get("confidence", 0.5)),
                    "reasoning": result.get("reasoning", "No reasoning provided")
                }
            except (json.JSONDecodeError, ValueError):
                logger.warning("Failed to parse LLM response, defaulting to ReAct agent")
                return {
                    "selected_agent": "react",
                    "confidence": 0.5,
                    "reasoning": "Failed to parse LLM response"
                }
                
        except Exception as e:
            logger.error(f"Error in agent selection: {e}")
            return {
                "selected_agent": "react",
                "confidence": 0.5,
                "reasoning": f"Error in selection: {str(e)}"
            }