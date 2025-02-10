from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
import tools.rag as rag
# from src.tools.rag import rag_with_reasoner

rag_with_reasoner = rag.rag_with_reasoner

def create_primary_agent() -> AssistantAgent:
    """Create and configure the primary agent."""
    global rag_with_reasoner

    return AssistantAgent(
        name="primary_agent",
        model_client=OpenAIChatCompletionClient(
            base_url="http://127.0.0.1:1234/v1",
            model="qwen2.5-7b-instruct-1m",
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": "unknown"
            }

            # api_key="YOUR_API_KEY",
        ),
        tools=[rag_with_reasoner],
    ) 