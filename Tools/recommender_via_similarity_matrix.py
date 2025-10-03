from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import numpy as np
import pickle
from scipy.sparse import load_npz, csr_matrix 
from typing import Any, List, Type
import pandas as pd 

# --- Utility Functions ---

def load_sparse_matrix(filename: str) -> csr_matrix:
    """Loads a sparse matrix from a file using scipy.sparse.load_npz."""
    return load_npz(filename)

def get_similar_books(book_index: int, simi_matrix: Any, n: int = 5) -> List[int]:
    """Finds the indices of the 'n' most similar books."""
    similarities = simi_matrix[book_index].toarray().flatten()
    similar_indices = np.argsort(similarities)[::-1]
    # Slice to skip the book itself (index 0) and take the next 'n' indices
    recommended_indices: List[int] = similar_indices[1:n+1].tolist()
    return recommended_indices

# --- Pydantic Input Schema ---

class RecommendationToolIN(BaseModel):
    # Only dynamic input needed for the agent call
    book_title: str = Field(..., description="The exact title of the book for which to find recommendations.")

# --- Class-based Tool Definition (THE FINAL TOOL) ---

class BookRecommendationTool(BaseTool):
    # Attributes that are filled upon instantiation in the main script
    book_list: pd.DataFrame 
    simi_matrix: Any # csr_matrix
    
    # Required LangChain Tool attributes
    name: str = "get_book_recommendations"
    description: str = "Finds a list of book titles similar to a given book title."
    args_schema: Type[BaseModel] = RecommendationToolIN
    
    def _run(self, book_title: str) -> str:
        """The core logic that runs when the agent calls the tool."""
        
        if 'Title' not in self.book_list.columns:
            return "Error: The book list provided must contain a column named 'Title'."
        
        try:
            match = self.book_list[self.book_list['Title'].str.lower() == book_title.lower()]
            if match.empty:
                raise IndexError
            book_index = match.index[0]

        except IndexError:
            return f"Error: Book title '{book_title}' not found in the book list."
            
        # Access the data via 'self'
        similar_indices = get_similar_books(book_index, self.simi_matrix, n=5) 
        recommended_titles: List[str] = self.book_list.iloc[similar_indices]['Title'].tolist()
        
        return f"Recommended books: {', '.join(recommended_titles)}"

# The main script must now import and instantiate BookRecommendationTool.

#-----------------------THIS CODE IS FOR TESTING ONLY---------------------
'''
similarity_matrix = load_sparse_matrix("/run/media/tst_imperial/Projects/BOOKai/large_flies/simi_sparse.npz")

with open("/run/media/tst_imperial/Projects/BOOKai/large_flies/books.pkl", "rb") as f:
    book_list = pickle.load(f)

BookRecommendation = BookRecommendationTool(book_list=book_list, simi_matrix=similarity_matrix)

recommendations = BookRecommendation._run("The New Face of Terrorism: Threats from Weapons of Mass Destruction")

print(recommendations)
'''