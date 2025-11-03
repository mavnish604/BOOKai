import uvicorn
import pickle
import pandas as pd
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List

# --- Import Your Project Modules ---
from bookai.api_models import (
    RecommendRequest, RecommendResponse, Recommendation,
    SummaryRequest, SummaryResponse
)
from bookai.config import BOOKS_PKL_PATH, SIMI_MATRIX_PATH

# --- Import your tools and router ---
from bookai.tools.similarity_tool import BookRecommendationTool, load_sparse_matrix
from bookai.vector_db.retriever import vector_store

# --- Import ALL our router functions ---
from bookai.router import (
    extract_title_from_query, 
    explain_recommendation, 
    format_final_response,
    get_book_summary_sync 
)
# --- Import the RAG agent function ---
from bookai.agents.rag_agent import get_rag_recommendations 

# -----------------------------------------------------------------
# 1. FastAPI Lifespan Event (Model Loading)
# -----------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's lifespan. Loads heavy models on startup.
    """
    print("--- Server is starting up... ---")
    
    print(f"Loading book list from: {BOOKS_PKL_PATH}")
    with open(BOOKS_PKL_PATH, "rb") as f:
        book_list_df = pd.read_pickle(BOOKS_PKL_PATH)
    
    print(f"Loading similarity matrix from: {SIMI_MATRIX_PATH}")
    simi_matrix = load_sparse_matrix(str(SIMI_MATRIX_PATH))
    
    print("Initializing Similarity Tool...")
    similarity_tool = BookRecommendationTool(
        book_list=book_list_df,
        simi_matrix=simi_matrix
    )
    
    book_titles_set = set(book_list_df['Title'].str.lower())
    
    app.state.book_titles_set = book_titles_set
    app.state.similarity_tool = similarity_tool
    app.state.vector_store = vector_store
    print("--- All assets loaded. Server is ready. ---")
    
    yield
    
    print("--- Server is shutting down... ---")

# -----------------------------------------------------------------
# 2. Initialize the FastAPI App
# -----------------------------------------------------------------
app = FastAPI(
    title="BOOKai API",
    description="API for hybrid book recommendations and summaries.",
    version="1.0.0",
    lifespan=lifespan
)

# --- CORS MIDDLEWARE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------
# 3. Define the API Endpoints
# -----------------------------------------------------------------

@app.post("/recommend", response_model=RecommendResponse)
async def post_recommendation(req: RecommendRequest, http_request: Request):
    """
    The main hybrid recommendation endpoint.
    """
    print(f"Received query: {req.query}")
    
    extracted = await extract_title_from_query(req.query)
    
    recommendation_list: List[Recommendation] = []
    final_response_str = ""
    
    # Check if a perfect title was found AND it exists in the fast matrix
    is_perfect_title = extracted.title_found and extracted.title and (extracted.title.lower() in http_request.app.state.book_titles_set)
    
    if is_perfect_title:
        # --- PATH A: FAST PATH (Similarity Matrix) ---
        print("--- Executing Fast Path (Similarity Matrix) ---")
        
        result_str = http_request.app.state.similarity_tool._run(extracted.title)
        
        if "Recommended books:" in result_str:
            titles = result_str.split("Recommended books: ")[1].split(", ")
            seen_titles = set()
            for title in titles:
                if title.lower() not in seen_titles:
                    seen_titles.add(title.lower())
                    rec_obj = Recommendation(
                        title=title, 
                        explanation="This book is structurally similar to your choice in our library.",
                        score=0.9 # High fixed score for matrix matches
                    )
                    recommendation_list.append(rec_obj)
        
    else:
        # --- PATH B: RAG PATH (Vector Store) ---
        # Use the original query if no title was found, or the extracted title if it wasn't a perfect matrix match (e.g., "Harry Potter" or "Fantasy")
        query_for_rag = extracted.title if extracted.title_found else req.query
        print(f"--- Executing RAG Path for: '{query_for_rag}' ---")
        
        try:
            # Run the blocking RAG agent in a separate thread
            rag_results = await asyncio.to_thread(get_rag_recommendations, query_for_rag)
            
            filtered_rag_results = []
            seen_titles = set()
            
            for title, score in rag_results:
                title_lower = title.lower()
                # Filtering logic: skip duplicates, original query, and single-word items (to filter out some names/junk)
                if title_lower in seen_titles or title_lower == query_for_rag.lower() or len(title.split()) <= 2:
                    continue
                seen_titles.add(title_lower)
                filtered_rag_results.append((title, score))

            for title, score in filtered_rag_results:
                # Retrieve the full document for the explanation function
                doc = http_request.app.state.vector_store.get(
                    where={"source_title": title},
                    include=["metadatas", "documents"]
                )
                doc_content = doc['documents'][0] if doc.get('documents') else "No description available."
                
                explanation = await explain_recommendation(
                    original_title=query_for_rag,
                    rec_title=title,
                    rec_doc=doc_content
                )
                
                recommendation_list.append(
                    Recommendation(
                        title=title, 
                        explanation=explanation,
                        score=score # Use the actual similarity score
                    )
                )
        except Exception as e:
            print(f"!!! RAG PATH ERROR: {e} !!!")
            pass 

    if not recommendation_list:
        final_response_str = "I'm sorry, I wasn't able to find any good recommendations for that. Could you try a different title or topic?"
    else:
        final_response_str = await format_final_response(req.query, recommendation_list)

    return RecommendResponse(
        response=final_response_str,
        recommendations=recommendation_list
    )

@app.post("/summary", response_model=SummaryResponse)
async def post_summary(request: SummaryRequest):
    """
    Generates a summary for a given book title using a web agent.
    """
    print(f"Received summary request for: {request.book_title}")
    
    try:
        # Call the synchronous function in a separate thread
        summary_text = await asyncio.to_thread(get_book_summary_sync, request.book_title)
        
        return SummaryResponse(
            book_title=request.book_title,
            summary=summary_text
        )
    except Exception as e:
        print(f"!!! SUMMARY ENDPOINT ERROR: {e} !!!")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

# -----------------------------------------------------------------
# 4. (Optional) Run the server directly for testing
# -----------------------------------------------------------------
if __name__ == "__main__":
    print("--- Starting Uvicorn server directly (for testing) ---")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)