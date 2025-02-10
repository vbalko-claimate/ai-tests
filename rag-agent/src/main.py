from agents.weather_agent import create_weather_agent
from agents.primary_agent import create_primary_agent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

import tools.rag as rag

async def init_rag() -> None:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"}
    )
    db_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
    vectordb = await rag.init_vectordb(persist_directory=db_dir, embedding_function=embeddings)

primary_agent = create_primary_agent()

async def main() -> None:
    await init_rag()

    

    # # Create primary agent
    # primary_agent = create_primary_agent()

    agent_team = RoundRobinGroupChat([primary_agent], max_turns=1)

    while True:
        user_input = input("Enter a message (type 'exit' to leave): ")
        if user_input.strip().lower() == "exit":
            break
        
        # Run the team and stream messages to the console
        stream = agent_team.run_stream(task=user_input)
        await Console(stream)




async def main_old() -> None:
    # Create weather agent


    weather_agent = create_weather_agent()

    # Define a team with a single agent and maximum auto-gen turns of 1
    agent_team = RoundRobinGroupChat([weather_agent], max_turns=1)

    while True:
        # Get user input from the console
        user_input = input("Enter a message (type 'exit' to leave): ")
        if user_input.strip().lower() == "exit":
            break
        
        # Run the team and stream messages to the console
        stream = agent_team.run_stream(task=user_input)# + " Dont forget to check the tool arguments names, types and other details in order to make valid call. Better double check it.")
        await Console(stream)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
