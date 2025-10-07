import os
import pickle
from typing import List

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from scipy.sparse import csr_matrix

# --- Local Imports ---
# These will work correctly when run from the BOOKAI root folder.
from logic import get_recommendation_flow
from Tools.recommender_via_similarity_matrix import BookRecommendationTool, load_sparse_matrix

# ======================================================================================
# 1. INITIAL SETUP & CONFIGURATION
# ======================================================================================
load_dotenv()

# --- Configuration from .env file ---
SIMILARITY_MATRIX_PATH = os.getenv("SIMILARITY_MATRIX_PATH")
BOOK_LIST_PATH = os.getenv("BOOK_LIST_PATH")

# --- Validate configuration ---
if not SIMILARITY_MATRIX_PATH or not BOOK_LIST_PATH:
    raise ValueError("SIMILARITY_MATRIX_PATH and BOOK_LIST_PATH must be set in the .env file")
if not os.getenv("GOOGLE_API_KEY") or not os.getenv("SERPER_API_KEY"):
    raise ValueError("GOOGLE_API_KEY and SERPER_API_KEY must be set in the .env file")


# ======================================================================================
# 2. LOAD MODELS & DATA (GLOBAL OBJECTS)
# ======================================================================================
print("Loading models and data... This may take a moment.")

try:
    similarity_matrix: csr_matrix = load_sparse_matrix(SIMILARITY_MATRIX_PATH)
    with open(BOOK_LIST_PATH, "rb") as f:
        book_list_df: pd.DataFrame = pickle.load(f)
    print("Successfully loaded similarity matrix and book list.")
except FileNotFoundError as e:
    print(f"FATAL ERROR: Could not load data files: {e}")
    # Exit if essential data is missing
    exit()

# --- Instantiate your recommendation tool ---
recommendation_tool = BookRecommendationTool(
    book_list=book_list_df,
    simi_matrix=similarity_matrix
)
print("BookRecommendationTool instantiated.")

# ======================================================================================
# 3. FASTAPI APPLICATION
# ======================================================================================
app = FastAPI(
    title="Book Recommender API",
    description="An API that recommends books based on user input, using modular, imported logic.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    # Allow all origins for development by using ["*"]
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for API Request/Response ---
class RecommendationRequest(BaseModel):
    prompt: str = Field(..., example="I really liked the book 'The Midnight Library', can you suggest something similar?")

class Book(BaseModel):
    title: str

class RecommendationResponse(BaseModel):
    source_book_title: str
    recommendations: List[Book]


# --- API Endpoints ---
@app.get("/", summary="Health Check")
def read_root():
    return {"status": "Book Recommender API is healthy and running."}


@app.post("/recommend", response_model=RecommendationResponse, summary="Get Book Recommendations")
async def recommend_books(request: RecommendationRequest):
    if not request.prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    try:
        result = await get_recommendation_flow(
            prompt=request.prompt,
            book_list_df=book_list_df,
            recommendation_tool=recommendation_tool
        )
        if not result["recommendations"]:
             raise HTTPException(
                 status_code=404,
                 detail=f"Could not find recommendations for '{result['source_book_title']}'."
             )
        return result
    except Exception as e:
        print(f"An unexpected error occurred in the API endpoint: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")


