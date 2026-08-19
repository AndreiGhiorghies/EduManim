from LLM.llm import LLM
from backend.agent.nodes.ffmpeg import make_ffmpeg_node
from backend.agent.state import AgentState
from backend.agent.nodes.scriptwriter import make_scriptwriter_node
from backend.agent.nodes.manim_coder import make_manim_coder_node
from backend.agent.nodes.tts import make_tts_node

import json

class SequentialGraph:

	def __init__(self, voice = 'narrator', video_quality="720p", callback_func=None):
		llm = LLM()

		self.callback_func = callback_func

		self.scriptwriter_node = make_scriptwriter_node(llm, max_retries=4)
		self.manim = make_manim_coder_node(llm, output_dir = "./output/video", max_retries=4)

		self.tts = make_tts_node(output_dir="./output/audio", voice=voice)
		self.assembler = make_ffmpeg_node(quality=video_quality)

	def invoke(self, state: AgentState) -> AgentState:
		if self.callback_func:
			self.callback_func("scripting", "Writing narration script...")
		state = self.scriptwriter_node(state)

		print("Scriptwriter Output:", state["scenes"], "\n\n")
		#state["scenes"] = state["scenes"][:1]

		if self.callback_func:
			self.callback_func("narration", "Synthesizing narration audio...", extra_data={"script_json": json.dumps(state.get("script"), ensure_ascii=False)})
		state = self.tts(state)
		print("After TTS Node: ", state, "\n\n", flush=True)

		if self.callback_func:
			self.callback_func("rendering", "Starting video rendering...")
		state = self.manim(state)

		if self.callback_func:
			self.callback_func("assembling", "Assembling final video...")
		state = self.assembler(state)

		print("STATE: ", state, "\n\n", flush=True)

		return state

def build_graph(voice, video_quality, callback_func = None):
	return SequentialGraph(voice.lower(), video_quality, callback_func=callback_func)