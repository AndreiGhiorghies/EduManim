from llama_cpp import Llama

class LLM_Local:
    def __init__(self, model_path: str = "./LLM/Models/qwen2.5-3b-instruct-q4_k_m.gguf"):
        
        """
            n_ctx=700,       # limit context size to avoid Out of Memory
        """
        self.llm = Llama(
            model_path=model_path,
            n_batch=256,
            n_threads=2,
            logits_all=True,
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