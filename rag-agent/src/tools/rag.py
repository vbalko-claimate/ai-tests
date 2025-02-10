
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.agents.reasoning_agent import create_reasoning_agent
# Global variable to hold the vector database instance
vectordb = None
print("vectordb module loaded")
# print(vectordb)


reasoner = None

async def init_reasoner() -> None:
    """
    Initialize the reasoner model.
    """
    global reasoner
    reasoner = create_reasoning_agent()


async def init_vectordb(persist_directory: str, embedding_function: HuggingFaceEmbeddings) -> Chroma:
    """
    Initialize a vector database from a directory of documents.
    """
    print("Initializing vector database...")
    global vectordb
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function)
    print("Vector database initialized")
    return vectordb

def save_response_to_markdown(response: str, file_path: str = "response.md") -> None:
    """
    Save the response message into a markdown file.
    
    Args:
        response (str): The text content to save.
        file_path (str): Path of the markdown file to create. Defaults to "response.md".
    """
    if isinstance(response, list):
        response = response[-1].content
    with open(file_path, "w", encoding="utf-8") as md_file:
        md_file.write(response)



async def rag_with_reasoner(user_query: str) -> str:
    """
    This is a RAG tool that takes in a user query and searches for relevant content from the vector database.
    The result of the search is given to a reasoning LLM to generate a response, so what you'll get back
    from this tool is a short answer to the user's question based on RAG context.

    Args:
        user_query: The user's question to query the vector database with.
    """
    global vectordb
    print("RAG with reasoner called")
    print(vectordb)
    if vectordb is None:
        raise ValueError("Vector database not initialized")
    
    # Search for relevant documents
    docs = vectordb.similarity_search(user_query, k=3)
    


    # Combine document contents
    context = "\n\n".join(doc.page_content for doc in docs)

    print("context: ", context)
    
    # Create prompt with context
    prompt = f"""Based on the following context, answer the user's question. Be concise and specific.
    If there isn't sufficient information, give as your answer a better query to perform RAG with.
    
Context:
{context}

Question: {user_query}

Answer:"""
    
    # Get response from reasoning model
    global reasoner
    print("reasoner: ", reasoner)
    if reasoner is None:
        print("reasoner not initialized, initializing...")
        await init_reasoner()
        print("reasoner initialized")
    response = await reasoner.run(task=prompt)
    save_response_to_markdown(response.messages)
    return response