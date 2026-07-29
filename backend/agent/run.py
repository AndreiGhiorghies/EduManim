from LLM.llm import LLM
from backend.agent.graph import build_graph
from backend.agent.state import create_initial_state


def run_agent():
    llm = LLM()
    
    app = build_graph(llm)
    
    test_query = "Explain how a llm works"
    initial_state = create_initial_state(user_query=test_query)
    
    result = app.invoke(initial_state)

    print("Script Output:", result.get("script"))
    
    return 0