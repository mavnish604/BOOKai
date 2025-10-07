import os
from typing import List, Dict, Any, Optional

from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
import pandas as pd

# --- Importing from YOUR modules ---
# These imports will now work correctly because main.py and logic.py
# are in the same root folder as RETRIVER and Tools.
from RETRIVER.fetch_from_VECTORDB import vector_store
from Tools.recommender_via_similarity_matrix import BookRecommendationTool
from Tools.web_search_tool import search_tool

# ======================================================================================
# 1. SETUP LLMS AND PARSERS
# ======================================================================================

# Initialize LLM (can be shared across functions)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

# --- Pydantic model for structured output from web search ---
# UPDATED: Made fields optional to handle cases where the LLM can't find all info.
class BookInfoParser(BaseModel):
    Title: Optional[str] = Field(None, description="The exact title of the book.")
    Author: Optional[str] = Field(None, description="The name(s) of the author(s).")
    Description: Optional[str] = Field(None, description="A brief, one-paragraph summary of the book.")
    Category: Optional[str] = Field(None, description="The primary genre or category of the book.")

pydantic_parser = PydanticOutputParser(pydantic_object=BookInfoParser)

# --- Agent for Web Search (when book is NOT in the database) ---
web_search_prompt_template = PromptTemplate(
    template="""
    You are a helpful assistant that gathers information about a specific book.
    Find the following details for the book titled '{title}': Title, Author, Description, and Category.
    If you cannot find a piece of information, you can leave it out.
    Format your final answer using the following instructions:
    {format_instructions}
    """,
    input_variables=["title"],
    partial_variables={"format_instructions": pydantic_parser.get_format_instructions()},
)

web_search_chain = web_search_prompt_template | llm | pydantic_parser

# ======================================================================================
# 2. CORE LOGIC FUNCTIONS (Implementing your diagram's flow)
# ======================================================================================

async def get_recommendations_from_vector_db(book_title: str) -> List[str]:
    """
    Handles the 'No' path of your diagram.
    """
    print(f"Book '{book_title}' not in database. Using web search and vector DB...")
    try:
        book_info = await web_search_chain.ainvoke({"title": book_title})

        # UPDATED: Safely build the query string, ignoring any missing fields.
        query_parts = []
        if book_info.Title:
            query_parts.append(f"Title: {book_info.Title}.")
        if book_info.Description:
            query_parts.append(f"Description: {book_info.Description}")
        if book_info.Author:
            query_parts.append(f"Author: {book_info.Author}.")
        if book_info.Category:
            query_parts.append(f"Category: {book_info.Category}.")
        
        if not query_parts:
             print("Could not find any information for the book to query the vector DB.")
             return []

        query_sentence = " ".join(query_parts)
        print(f"Constructed VectorDB Query: {query_sentence}")

        results = await vector_store.asimilarity_search_with_relevance_scores(query=query_sentence, k=5)
        return [doc.metadata.get('source_title', 'Unknown Title') for doc, score in results]
    except Exception as e:
        print(f"Error during vector DB recommendation flow: {e}")
        return []

def get_recommendations_from_similarity_matrix(book_title: str, recommendation_tool: BookRecommendationTool) -> List[str]:
    """
    Handles the 'Yes' path of your diagram.
    """
    print(f"Book '{book_title}' found in database. Using similarity matrix...")
    try:
        result_str = recommendation_tool._run(book_title)
        if "Error:" in result_str:
            print(result_str)
            return []
        titles_part = result_str.replace("Recommended books:", "").strip()
        return [title.strip() for title in titles_part.split(',')]
    except Exception as e:
        print(f"Error during similarity matrix recommendation flow: {e}")
        return []

# ======================================================================================
# 3. MAIN ORCHESTRATION FUNCTION
# ======================================================================================

async def get_recommendation_flow(
    prompt: str,
    book_list_df: pd.DataFrame,
    recommendation_tool: BookRecommendationTool
) -> Dict[str, Any]:
    """Orchestrates the entire recommendation process."""
    title_extraction_prompt = PromptTemplate.from_template(
        "Extract the exact book title from the following query. Return only the title.\n\nQuery: {prompt}\n\nTitle:"
    )
    title_chain = title_extraction_prompt | llm | StrOutputParser()
    book_title = await title_chain.ainvoke({"prompt": prompt})
    book_title = book_title.strip().strip('"')
    print(f"Extracted book title: '{book_title}'")

    is_in_db = not book_list_df[book_list_df['Title'].str.lower() == book_title.lower()].empty

    if is_in_db:
        recommended_titles = get_recommendations_from_similarity_matrix(book_title, recommendation_tool)
    else:
        recommended_titles = await get_recommendations_from_vector_db(book_title)

    return {
        "source_book_title": book_title,
        "recommendations": [{"title": title} for title in recommended_titles]
    }

