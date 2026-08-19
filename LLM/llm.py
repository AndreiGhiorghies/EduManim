import os
from dotenv import load_dotenv

load_dotenv()

class LLM:
    _instance = None

    # Make it singleton to ensure only one instance of the llm is loaded into memory
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LLM, cls).__new__(cls)
            cls._instance._initialized = False
        
        return cls._instance
    
    def __init__(self):
        if getattr(self, '_initialized', False):
            return
        
        route = os.getenv("ROUTE") or ""
        if route == "":
            raise ValueError("ROUTE is not set in the environment variables. Acceptable values are 'LOCAL' or 'REMOTE'.")
        
        if route.upper() == "LOCAL":
            from LLM.llm_local import LLM_Local
            self.llm = LLM_Local()
        elif route.upper() == "REMOTE":
            from LLM.llm_remote import LLM_Remote
            self.llm = LLM_Remote()
        else:
            raise ValueError("Invalid ROUTE value. Acceptable values are 'LOCAL' or 'REMOTE'.")

        self._initialized = True

    def generate(self, prompt, system_instruction: str = "You are a helpful assistant.", number_of_attempts: int = 2, temperature: float = 0.0, max_tokens: int = 4096) -> str:

        print("Made LLM call with prompt:", prompt, "\n\n")

        ans = self.llm.generate(prompt = prompt, system_instruction = system_instruction, number_of_attempts = number_of_attempts, temperature = temperature, max_tokens = max_tokens)

        print("LLM call returned:", ans, "\n\n")

        return ans