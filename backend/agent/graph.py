from backend.agent.state import AgentState
from tools.scriptwriter import make_scriptwriter_node

class SequentialGraph:

	def __init__(self, llm):
		self.scriptwriter_node = make_scriptwriter_node(llm, max_retries=2)

	def invoke(self, state: AgentState) -> AgentState:
	
		state = self.scriptwriter_node(state)

		return state

def build_graph(llm):
	return SequentialGraph(llm)