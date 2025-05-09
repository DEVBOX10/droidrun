from ..llm.client import LLMClient
from ..llm.providers.openai import OpenAIProvider
from ..graph import parse_graph, Graph, State
from typing import Tuple, Dict, Optional, List, TypedDict
from collections import defaultdict
from ..prompts import Prompt
import logging
import re
import os

logger = logging.getLogger(__name__)


class PlanInfo(TypedDict):
    goal_plan: str
    num_input_tokens_plan: int
    num_output_tokens_plan: int
    goal_plan_cost: float


class GraphInfo(TypedDict):
    graph: str
    num_input_tokens_dag: int
    num_output_tokens_dag: int
    graph_cost: float


class PlannerInfo(PlanInfo, GraphInfo, TypedDict):
    pass


class Planner:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        self.generator_llm = LLMClient(
            provider=OpenAIProvider(model="gpt-4o", api_key=api_key),
            system_prompt=Prompt("plan_generator").build(),
        )
        self.graph_translator_llm = LLMClient(
            provider=OpenAIProvider(model="gpt-4o", api_key=api_key),
            system_prompt=Prompt("graph_translator").build(),
        )
        self.episode_summarizer_llm = LLMClient(
            provider=OpenAIProvider(model="gpt-4o", api_key=api_key),
            system_prompt=Prompt("episode_summarizer").build(),
        )

        self.history: List[str] = []
        self.turn_count = 0

    async def summarize_episode(self, trajectory):
        """Summarize the episode experience for lifelong learning reflection
        Args:
            trajectory: str: The episode experience to be summarized
        """

        # Create Reflection on whole trajectories for next round trial, keep earlier messages as exemplars
        self.episode_summarizer_llm.add_message(trajectory, role="user")
        subtask_summarization = await self.episode_summarizer_llm.get_response()
        self.episode_summarizer_llm.add_message(subtask_summarization, role="assistant")

        return subtask_summarization

    async def _generate_plan(
        self,
        observation: Dict,
        instruction: str,
        failed_subtask: Optional[State] = None,
        completed_subtasks_list: List[State] = [],
        remaining_subtasks_list: List[State] = [],
    ) -> Tuple[PlanInfo, str]:
        logger.info("GENERATING PLAN: %s", instruction)

        # Converts a list of DAG Nodes into a natural langauge list
        def format_subtask_list(subtasks: List[State]) -> str:
            res = ""
            for idx, node in enumerate(subtasks):
                res += f"{idx+1}. **{node.name}**:\n"
                bullets = re.split(r"(?<=[.!?;]) +", node.info)
                for bullet in bullets:
                    res += f"   - {bullet}\n"
                res += "\n"
            return res

        self.generator_llm.add_system_prompt(
            self.generator_llm.system_prompt.replace("TASK_DESCRIPTION", instruction)
        )

        if failed_subtask:
            generator_message = (
                f"The subtask {failed_subtask} cannot be completed. Please generate a new plan for the remainder of the trajectory.\n\n"
                f"Successfully Completed Subtasks:\n{format_subtask_list(completed_subtasks_list)}\n"
            )
        elif len(completed_subtasks_list) + len(remaining_subtasks_list) > 0:
            generator_message = (
                "The current trajectory and desktop state is provided. Please revise the plan for the following trajectory.\n\n"
                f"Successfully Completed Subtasks:\n{format_subtask_list(completed_subtasks_list)}\n"
                f"Future Remaining Subtasks:\n{format_subtask_list(remaining_subtasks_list)}\n"
            )
        # Initial plan case
        else:
            generator_message = "Please generate the initial plan for the task.\n"

        logger.info("GENERATOR MESSAGE: %s", generator_message)

        # TODO: add observation a11y info
        self.generator_llm.add_message(
            generator_message,
            image_content=observation.get("screenshot", None),
            role="user",
        )
        plan = await self.generator_llm.get_response()
        self.generator_llm.add_message(plan, role="assistant")
        self.history.append(plan)
        self.turn_count += 1

        # Set Cost based on GPT-4o
        input_tokens, output_tokens = self.generator_llm.calculate_tokens()
        logger.info("INPUT TOKENS: %s", input_tokens)
        logger.info("OUTPUT TOKENS: %s", output_tokens)
        cost = self.generator_llm.provider.calculate_cost(input_tokens, output_tokens)
        logger.info("COST: %s", cost)
        planner_info = PlanInfo(
            goal_plan=plan,
            num_input_tokens_plan=input_tokens,
            num_output_tokens_plan=output_tokens,
            goal_plan_cost=cost,
        )

        logger.info("PLAN: %s", plan)
        assert type(plan) == str

        return planner_info, plan

    async def _generate_graph(
        self, instruction: str, plan: str
    ) -> Tuple[GraphInfo, Graph]:
        # For the re-planning case, remove the prior input since this should only translate the new plan
        self.graph_translator_llm.reset()

        # Add initial instruction and plan to the agent's message history
        self.graph_translator_llm.add_message(
            f"Instruction: {instruction}\nPlan: {plan}", role="user"
        )

        logger.info("GENERATING GRAPH")

        # Generate DAG
        graph_raw = await self.graph_translator_llm.get_response()
        graph = parse_graph(graph_raw)

        logger.info("Generated Graph: %s", graph_raw)

        self.graph_translator_llm.add_message(graph_raw, role="assistant")

        input_tokens, output_tokens = self.graph_translator_llm.calculate_tokens()
        cost = self.graph_translator_llm.provider.calculate_cost(
            input_tokens, output_tokens
        )

        graph_info = GraphInfo(
            graph=graph_raw,
            num_input_tokens_dag=input_tokens,
            num_output_tokens_dag=output_tokens,
            graph_cost=cost,
        )

        logger.info("GRAPH: %s", graph_raw)
        assert type(graph) == Graph

        return graph_info, graph

        # def _topological_sort(self, graph: Graph) -> List[Node]:
        """Topological sort of the DAG using DFS
        dag: Dag: Object representation of the DAG with nodes and edges
        """

    """
        def dfs(node_name, visited, stack):
            visited[node_name] = True
            for neighbor in adj_list[node_name]:
                if not visited[neighbor]:
                    dfs(neighbor, visited, stack)
            stack.append(node_name)

        # Convert edges to adjacency list
        adj_list = defaultdict(list)
        for u, v in graph.edges:
            adj_list[u.name].append(v.name)

        visited = {node.name: False for node in graph.nodes}
        stack = []

        for node in graph.nodes:
            if not visited[node.name]:
                dfs(node.name, visited, stack)

        # Return the nodes in topologically sorted order
        sorted_nodes = [
            next(n for n in graph.nodes if n.name == name) for name in stack[::-1]
        ]
        return sorted_nodes"""

    async def get_state_machine_graph(
        self,
        instruction: str,
        observation: Dict,
        failed_subtask: Optional[State] = None,
        completed_subtasks_list: List[State] = [],
        remaining_subtasks_list: List[State] = [],
    ) -> Tuple[PlannerInfo, Graph]:
        """Generate the action list based on the instruction
        instruction:str: Instruction for the task
        """

        planner_info, plan = await self._generate_plan(
            observation,
            instruction,
            failed_subtask,
            completed_subtasks_list,
            remaining_subtasks_list,
        )

        # Generate the DAG
        graph_info, graph = await self._generate_graph(instruction, plan)

        # Topological sort of the DAG
        # action_queue = self._topological_sort(graph)

        return PlannerInfo(**planner_info, **graph_info), graph
