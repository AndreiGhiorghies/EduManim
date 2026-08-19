from llama_cpp import Llama

class LLM_Local:
    def __init__(self, model_path: str = "./LLM/Models/qwen2.5-32b-instruct-q8_0-00001-of-00009.gguf"):
        
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=8192,
            n_batch=256,
            n_threads=4,
            verbose=False
        )

    def generate(self, prompt: str, system_instruction: str, number_of_attempts: int, temperature: float, max_tokens: int) -> str:
        messages = [
            {
                "role": "system", 
                "content": system_instruction
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9
        )

        return response["choices"][0]["message"]["content"]