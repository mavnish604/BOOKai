#-------------nessary imports-------------------------------
from langchain_community.utilities import GoogleSerperAPIWrapper
from dotenv import load_dotenv
from langchain_core.tools import Tool

#-----------------setting up the enviroment---------------------
import os
load_dotenv()
#----------------------Setting up the search tool------------------
api_key=os.getenv("SERPER_API_KEY")

search = GoogleSerperAPIWrapper()

search_tool = Tool(
    func=search.run,
    name="search_tool",
    description="this tool will be used to scrape book data from the web"
)
#--------------------test-------------------------------------
#print(search_tool.invoke("who was the first pm of india"))
