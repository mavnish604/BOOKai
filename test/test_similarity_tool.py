# test/test_similarity_tool.py
import pickle
import sys
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

# --- Add Project Root to Python Path ---
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))
# -------------------------------------

# Import from our new, correct locations
from bookai.tools.similarity_tool import BookRecommendationTool, load_sparse_matrix
from bookai.config import BOOKS_PKL_PATH, SIMI_MATRIX_PATH

# --- Load Data (using config paths) ---
print(f"Loading similarity matrix from {SIMI_MATRIX_PATH}...")
similarity_matrix = load_sparse_matrix(str(SIMI_MATRIX_PATH))

print(f"Loading book list from {BOOKS_PKL_PATH}...")
with open(BOOKS_PKL_PATH, "rb") as f:
    book_list = pickle.load(f)
# -----------------------------

print("Instantiating recommendation tool...")
recommendation_tool_instance = BookRecommendationTool(
    book_list=book_list,
    simi_matrix=similarity_matrix
)
# -------------------------------------

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash") # Using 1.5-flash
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm=model,
    tools=[recommendation_tool_instance],
    prompt=prompt
)

agent_excutor = AgentExecutor(
    agent=agent,
    tools=[recommendation_tool_instance],
    verbose=True
)

#----------------------TEST--------------------------------------------------------

'''
Title = "The New Face of Terrorism: Threats from Weapons of Mass Destruction"
parser = StrOutputParser()

# --- Agent Invocation ---
print(f"Running agent for book: '{Title}'")
result = agent_excutor.invoke({"input": f"""You are an helpful libray agent,
                            Your job is to recommed people books and tell tp them 
                            in a very repectful way,
                            user liked {Title} recommed him some books in a good format"""})

print("\n--- Agent Output ---")
print(parser.invoke(result["output"]))
'''