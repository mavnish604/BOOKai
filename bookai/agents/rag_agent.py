import sys
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from pydantic import BaseModel, Field

# --- Corrected Imports ---
from bookai.tools.web_search_tool import search_tool
from bookai.vector_db.retriever import vector_store
# ---------------------------

load_dotenv()

# --- Setup (These objects are created once and reused) ---
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm=model,
    prompt=prompt,
    tools=[search_tool]
)
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool],
    verbose=False,  # Set to True for detailed agent logs
    handle_parsing_errors=True
)

class AgentOutputparser(BaseModel):
    Title: str = Field(..., description="the title of the book")
    Author: str = Field(..., description="the name of the author")
    Description: str = Field(..., description="A brief description of the book")
    category: str = Field(..., description="Category of the book")
    publisher: str = Field(..., description="name of the publisher")

parser = PydanticOutputParser(pydantic_object=AgentOutputparser)

template = PromptTemplate(
    template="""You are a helpful web agent. Your goal is to gather data about the specific book: {title}.
Format the output in the following way: {format_in}
The data should contain: title, author name, description, category, publisher name, etc.
""",
    input_variables=["title"],
    partial_variables={"format_in": parser.get_format_instructions()}
)

# --- The New Modular Function ---

def get_rag_recommendations(book_title: str) -> list[tuple[str, float]]:
    """
    Gets book recommendations using the RAG agent (Web Search + Vector Query).
    
    Args:
        book_title: The title of the book to get recommendations for.
        
    Returns:
        A list of (title, score) tuples for the recommendations.
    """
    print(f"\n--- RAG Agent: Starting for '{book_title}' ---")
    
    # -----------------------------------------------------------------
    # Step 1: Run the agent to get book info from the web
    # -----------------------------------------------------------------
    print("RAG Step 1: Searching web for book details...")
    prompt_text = template.format(title=book_title)
    result = agent_executor.invoke({"input": prompt_text})

    try:
        parsed = parser.invoke(result["output"])
    except Exception as e:
        print(f"Error parsing agent output: {e}")
        print(f"Raw output was: {result.get('output', 'N/A')}")
        return []  # Return empty list on failure

    # -----------------------------------------------------------------
    # Step 2: Use the parsed title to query the vector store
    # -----------------------------------------------------------------
    print("RAG Step 2: Querying vector store...")
    
    # We use the simple, working query
    query_sentence = parsed.Title 

    try:
        res = vector_store.similarity_search_with_relevance_scores(
            query=query_sentence,
            k=5  # Get 5 recommendations
        )
    except Exception as e:
        print(f"Error querying vector store: {e}")
        return []

    # -----------------------------------------------------------------
    # Step 3: Format and return the results
    # -----------------------------------------------------------------
    print("RAG Step 3: Formatting results...")
    recommendations = []
    for doc, score in res:
        title = doc.metadata.get('source_title', 'TITLE NOT FOUND')
        # Skip the book if it's the one we searched for
        if title.lower() != book_title.lower():
            recommendations.append((title, score))
            
    return recommendations

# -----------------------------------------------------------------
# This block is now just for testing the function directly
# -----------------------------------------------------------------
if __name__ == "__main__":
    
    print("--- Testing RAG Agent Function ---")

    # --- Test Case 1: Harry Potter (Should find matches) ---
    title_1 = "Harry Potter and the Half Blood Prince"
    recommendations_1 = get_rag_recommendations(title_1)
    
    print(f"\n--- Recommendations for '{title_1}' ---")
    if recommendations_1:
        for title, score in recommendations_1:
            print("{:<70} | {:<10.4f}".format(title[:70], score))
    else:
        print("No recommendations found.")
    print("-" * 82)

    # --- Test Case 2: Germs (Should find matches) ---
    title_2 = "Germs : Biological Weapons and America's Secret War"
    recommendations_2 = get_rag_recommendations(title_2)

    print(f"\n--- Recommendations for '{title_2}' ---")
    if recommendations_2:
        for title, score in recommendations_2:
            print("{:<70} | {:<10.4f}".format(title[:70], score))
    else:
        print("No recommendations found.")
    print("-" * 82)