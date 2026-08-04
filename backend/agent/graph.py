from backend.agent.nodes.ffmpeg import make_ffmpeg_node
from backend.agent.state import AgentState
from backend.agent.nodes.scriptwriter import make_scriptwriter_node
from backend.agent.nodes.manim_coder import process_all_scenes
from backend.agent.nodes.tts import make_tts_node
import asyncio

class SequentialGraph:

	def __init__(self, llm):
		self.scriptwriter_node = make_scriptwriter_node(llm, max_retries=2)
		self.tts = make_tts_node(output_dir=f"./output/audio", voice="narrator")
		self.assembler = make_ffmpeg_node(f"./output/final.mp4", quality="720p")
		self.llm = llm

	def invoke(self, state: AgentState) -> AgentState:
	
		state = self.scriptwriter_node(state)

		print("Scriptwriter Output:", state["scenes"], "\n\n")

		state = asyncio.run(self.tts(state))

		state = asyncio.run(process_all_scenes(state, llm = self.llm, output_dir="output", max_retries=2))

		state = self.assembler(state)

		print("STATE: ", state)

		return state

def build_graph(llm):
	return SequentialGraph(llm)