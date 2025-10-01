#!/run/media/tst_imperial/Projects/BOOKAI/venv/bin/python
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
import numpy as np
import pickle
from scipy.sparse import load_npz, csr_matrix # load_npz is imported from scipy.sparse
from typing import Any, List
import pandas as pd 

# --- Utility Functions ---

def load_sparse_matrix(filename: str) -> csr_matrix:
    """Loads a sparse matrix from a file using scipy.sparse.load_npz."""
    return load_npz(filename)

# --- Pydantic Input Schema ---

class RecommendationToolIN(BaseModel):
    # This is the user input for the book title
    book_title: str = Field(..., description="The exact title of the book for which to find recommendations.")
    
    # These parameters are typically loaded in the environment
    book_list: Any = Field(..., description="The Pandas DataFrame containing all book titles and their indices.")
    simi_matrix: Any = Field(..., description="The loaded sparse similarity matrix (scipy.sparse.csr_matrix).") 

# --- Helper Function ---

def get_similar_books(book_index: int, simi_matrix: Any, n: int = 5) -> List[int]:
    """Finds the indices of the 'n' most similar books."""
    
    # 1. Get the similarity row, convert to dense array, and flatten
    similarities = simi_matrix[book_index].toarray().flatten()
    
    # 2. Get indices for sorting in descending order (most similar first)
    similar_indices = np.argsort(similarities)[::-1]
    
    # 3. Slice to skip the book itself (index 0) and take the next 'n' indices
    recommended_indices: List[int] = similar_indices[1:n+1].tolist()
    
    return recommended_indices

# --- Actual Tool Function (`recon`) ---

def recon(book_title: str, book_list: pd.DataFrame, simi_matrix: Any) -> List[str]:
    """
    TOOL: Finds and returns the titles of similar books for a given book title.
    
    The function relies on a pre-loaded book_list (DataFrame) and simi_matrix (csr_matrix).
    """
    
    # 1. Find the index of the target book
    try:
        # Perform a case-insensitive match for robustness
        book_index = book_list[book_list['Title'].str.lower() == book_title.lower()].index[0]
    except IndexError:
        return [f"Error: Book title '{book_title}' not found in the book list."]
        
    # 2. Get the indices of the similar books (default to 5 recommendations)
    similar_indices = get_similar_books(book_index, simi_matrix, n=5) 
    
    # 3. Use the indices to fetch the actual book titles
    recommended_titles: List[str] = book_list.iloc[similar_indices]['Title'].tolist()
    
    return recommended_titles

# --- Structured Tool Definition ---

recommendation_tool = StructuredTool.from_function(
    func=recon,
    name="get_book_recommendations",
    description="Finds a list of book titles similar to a given book title using a sparse similarity matrix and a book list.",
    args_schema=RecommendationToolIN,
)

#-----------------------THIS CODE IS FOR TESTING ONLY---------------------

'''
with open("/run/media/tst_imperial/Projects/BOOKAI/large_flies/books.pkl","rb") as f:
    book_list=pickle.load(f)

simi_matrix = load_sparse_matrix("/run/media/tst_imperial/Projects/BOOKAI/large_flies/simi_sparse.npz")

recommendations=recon("The New Face of Terrorism: Threats from Weapons of Mass Destruction", book_list,simi_matrix)

print(recommendations)
'''