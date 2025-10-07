import sys
import os

# Command to add the project root to Python's path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ... keep the rest of your imports and code below this


from Tools.web_search_tool import search_tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent,AgentExecutor
from pydantic import BaseModel,Field
from RETRIVER.fetch_from_VECTORDB import vector_store

# ...existing imports...

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm=model,
    prompt=prompt,
    tools=[search_tool]
)
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool],
    verbose=False,
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



#__________________________this part of the code shoulf only be touched while testing_____________________
'''
title = "Harry Potter and the Half Blood Prince"
prompt_text = template.format(title=title)

# Run the agent_executor with the constructed prompt
result = agent_executor.invoke({"input": prompt_text})

# Parse the output using your parser
parsed = parser.invoke(result["output"])

print(parsed)
'''
#--------Searching the vector db----------------------------

title = "Harry Potter and the Half Blood Prince"
prompt_text = template.format(title=title)

# Run the agent_executor with the constructed prompt
result = agent_executor.invoke({"input": prompt_text})

#print(result)
# Parse the output using your parser
parsed = parser.invoke(result["output"])


query_sentence = (
    f"{parsed.Title}: {parsed.Description} "
    f"Written by {parsed.Author} and published by {parsed.publisher}. "
    f"Category: {parsed.category}."
)


res=result=vector_store.similarity_search_with_relevance_scores(
    query=query_sentence,
    k=3
)

for doc, score in res:
    # 1. Access the 'source_title' key from the Document's metadata dictionary
    title = doc.metadata.get('source_title', 'TITLE NOT FOUND')
    
    # 2. Print the formatted output
    # Use slicing [:] to ensure the title doesn't overflow if it's too long
    print("{:<70} | {:<10.4f}".format(title[:70], score)) 

print("-" * 82)
