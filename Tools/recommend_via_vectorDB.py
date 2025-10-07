from langchain_chroma import Chroma
from web_search_tool import search_tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent,AgentExecutor
from pydantic import BaseModel,Field

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
    verbose=True,
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

