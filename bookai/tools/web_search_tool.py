import sys
from pathlib import Path
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import Tool

# --- Add Project Root to Python Path ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
# -------------------------------------

from bookai.config import SERPER_API_KEY

if not SERPER_API_KEY:
    print("SERPER_API_KEY not found in .env file.")
    # You might want to raise an exception here
    
search = GoogleSerperAPIWrapper(serper_api_key=SERPER_API_KEY)

search_tool = Tool(
    func=search.run,
    name="search_tool",
    description="this tool will be used to scrape book data from the web"
)