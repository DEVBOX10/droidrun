from .agents.planner import Planner


class DroidRunAgent:
    def __init__(self):
        pass

    async def run(self, instruction: str, max_retries: int = -1) -> bool:
        planner = Planner()
        info, graph = await planner.get_state_machine_graph(instruction, {})
        print(info)
        print(graph)

        for _ in range(max_retries):
            # supervisor bekommt graph (state maschine) 
            # supervisor entscheidet welche pfad gegangen werden soll basierend auf observation und subtask goal
            # supervisor gibt codeact spezialisierte tools & context
            # codeact führt action aus

            # supervisor reflected ob er sich richtig entschieden hat.
            # wenn nicht muss planner neuen graphen generieren
            # ansonsten gehts weiter
            pass

        return True


if __name__ == "__main__":
    agent = DroidRunAgent()
    agent.run("Create a new folder called 'test'")
