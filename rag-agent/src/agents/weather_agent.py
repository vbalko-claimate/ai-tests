from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from src.tools.weather import get_weather

def create_weather_agent() -> AssistantAgent:
    """Create and configure the weather agent."""
    return AssistantAgent(
        name="weather_agent",
        model_client=OpenAIChatCompletionClient(
            base_url="http://127.0.0.1:1234/v1",
            model="deepseek-r1-distill-qwen-1.5b",
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": "unknown"
            }

            # api_key="YOUR_API_KEY",
        ),
        tools=[get_weather],
    ) 