from recommender_via_similarity_matrix import BookRecommendationTool, load_sparse_matrix 
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import pickle
from langchain.agents import create_react_agent,AgentExecutor
from langchain import hub
from recommender_via_similarity_matrix import load_sparse_matrix
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# --- Load Data  ---
similarity_matrix=load_sparse_matrix("/run/media/tst_imperial/Projects/BOOKai/large_flies/simi_sparse.npz")

with open("/run/media/tst_imperial/Projects/BOOKai/large_flies/books.pkl","rb") as f:
    book_list=pickle.load(f)
# -----------------------------

# Instantiate the class, passing the large data objects here.
recommendation_tool_instance = BookRecommendationTool(
    book_list=book_list,
    simi_matrix=similarity_matrix
)
# -------------------------------------

model=ChatGoogleGenerativeAI(model="gemini-2.0-flash")
prompt=hub.pull("hwchase17/react")

agent=create_react_agent(
    llm=model,
    # Use the instance
    tools=[recommendation_tool_instance],
    prompt=prompt
)

agent_excutor=AgentExecutor(
    agent=agent,
    # Use the instance
    tools=[recommendation_tool_instance],
    verbose=True
)

Title="The New Face of Terrorism: Threats from Weapons of Mass Destruction"
parser=StrOutputParser()
# --- Agent Invocation (Clean Input) ---
# The input is clean because the data is now stored in the tool instance.
result = agent_excutor.invoke({"input":f"""You are an helpful libray agent,
                            Your job is to recommed people books and tell tp them 
                            in a very repectful way,
                            user liked {Title} recommed him some books in a good format"""})
print(parser.invoke(result["output"] if isinstance(result, dict) and "output" in result else result))