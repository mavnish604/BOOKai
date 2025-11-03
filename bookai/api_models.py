from pydantic import BaseModel, Field
from typing import Optional, List

# -----------------------------------------------------------------
# Models for: POST /recommend
# -----------------------------------------------------------------

class RecommendRequest(BaseModel):
    """
    The input JSON object for the /recommend endpoint.
    """
    query: str = Field(..., 
                       description="The user's full natural language query.", 
                       example="I really liked 'Harry Potter', got any ideas?")

class Recommendation(BaseModel):
    """
    A single book recommendation.
    """
    title: str = Field(..., description="The title of the recommended book.")
    explanation: Optional[str] = Field(None, 
                                       description="The AI-generated reason *why* this book is recommended.")
    # --- ADDED FIELD ---
    score: float = Field(..., description="The similarity score (chance of liking).")

class RecommendResponse(BaseModel):
    """
    The output JSON object for the /recommend endpoint.
    """
    response: str = Field(..., 
                          description="The final, friendly, conversational response from the AI.",
                          example="Based on 'Harry Potter', you might also enjoy these titles...")
    recommendations: List[Recommendation] = Field(..., 
                                                description="The list of recommended books.")

# -----------------------------------------------------------------
# Models for: POST /summary
# -----------------------------------------------------------------

class SummaryRequest(BaseModel):
    """
    The input JSON object for the /summary endpoint.
    """
    book_title: str = Field(..., 
                            description="The exact title of the book to summarize.",
                            example="Germs : Biological Weapons and America's Secret War")

class SummaryResponse(BaseModel):
    """
    The output JSON object for the /summary endpoint.
    """
    book_title: str = Field(..., description="The title of the book that was summarized.")
    summary: str = Field(..., description="The AI-generated summary.")

# -----------------------------------------------------------------
# Models for: The Hybrid Router Logic
# -----------------------------------------------------------------

class ExtractedTitle(BaseModel):
    """
    The structured output from our title extraction chain.
    """
    title_found: bool = Field(..., 
                             description="True if a book title was confidently extracted.")
    title: Optional[str] = Field(None, 
                                 description="The exact book title, if found.",
                                 example="Harry Potter and the Half Blood Prince")
    reasoning: str = Field(..., 
                           description="A brief explanation from the AI on why it found or didn't find a title.")