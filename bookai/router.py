# bookai/router.py
import os
import httpx 
import asyncio
import json 
from typing import List, AsyncIterator 

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from pydantic import ValidationError
from dotenv import load_dotenv

# --- Import your project modules ---
from bookai.config import (
    GROQ_API_KEY, GOOGLE_API_KEY, SERPER_API_KEY, HF_TOKEN
)
from bookai.api_models import ExtractedTitle, Recommendation
from bookai.tools.web_search_tool import search_tool
from bookai.agents.rag_agent import get_rag_recommendations 

load_dotenv()
# -----------------------------------------------------------------
# 1. Initialize API Clients
# -----------------------------------------------------------------
API_KEYS = [
    ("GROQ_API_KEY", GROQ_API_KEY), ("GOOGLE_API_KEY", GOOGLE_API_KEY),
    ("SERPER_API_KEY", SERPER_API_KEY), ("HF_TOKEN", HF_TOKEN)
]
missing_keys = [name for name, value in API_KEYS if not value or value.strip() == ""]
if missing_keys:
    # This check is vital for stability
    raise ValueError(f"CRITICAL ERROR: The following API keys are missing or empty in your .env file: {', '.join(missing_keys)}")

# --- Tier 1: Groq Client (Fast tasks) ---
groq_chat_model = ChatOpenAI(
    model_name="openai/gpt-oss-20b",
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
    temperature=0
)

# --- Tier 2: Gemini Client (Powerful tasks) ---
gemini_pro_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)

# --- Tier 3: Hugging Face Backup ---
HF_MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1" # Changed to reliable Mixtral for stability
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# --- SYNCHRONOUS HF API CALL (For compatibility with sync thread) ---
def call_hf_api_sync(prompt: str) -> str:
    """
    Calls the Hugging Face Inference API as a synchronous backup.
    """
    print(f"--- Failing over to Hugging Face model: {HF_MODEL_ID} (Sync) ---")
    try:
        # Use synchronous httpx.Client for compatibility with asyncio.to_thread
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                HF_API_URL, 
                headers=HF_HEADERS, 
                json={"inputs": prompt, "parameters": {"max_new_tokens": 1024, "temperature": 0.7, "return_full_text": False}}
            )
            response.raise_for_status() 
            result = response.json()
            if isinstance(result, list) and "generated_text" in result[0]:
                return result[0]["generated_text"]
            else:
                return "Backup model returned an unexpected format."

    except Exception as e:
        print(f"!!! HUGGING FACE BACKUP FAILED: {e} !!!")
        return "Primary AI model failed, and the backup model is also unavailable."

# -----------------------------------------------------------------
# 2. Chains & Agents (Definitions)
# -----------------------------------------------------------------

# === Title Extraction Chain ===
title_parser = JsonOutputParser(pydantic_object=ExtractedTitle)
title_extraction_prompt = ChatPromptTemplate.from_template("...")
title_extraction_chain = title_extraction_prompt | groq_chat_model | title_parser

# === Summarization Agent & Executor ===
react_prompt = hub.pull("hwchase17/react")
summary_web_agent = create_react_agent(llm=gemini_pro_model, prompt=react_prompt, tools=[search_tool])
summary_agent_executor = AgentExecutor(agent=summary_web_agent, tools=[search_tool], verbose=False, handle_parsing_errors=True)


# -----------------------------------------------------------------
# 3. Router Functions (Callable by main.py)
# -----------------------------------------------------------------

async def extract_title_from_query(query: str) -> ExtractedTitle:
    """Uses a fast LLM (Groq) to extract a single book title from a user's query."""
    print(f"--- Groq: Extracting title from query: '{query}' ---")
    try:
        response = await title_extraction_chain.ainvoke({"query": query})
        return ExtractedTitle.model_validate(response)
    except (ValidationError, Exception) as e:
        print(f"!!! ROUTER ERROR (Title Extraction): {e} !!!")
        return ExtractedTitle(title_found=False, reasoning=f"An API or parsing error occurred: {e}")

async def explain_recommendation(original_title: str, rec_title: str, rec_doc: str) -> str:
    """Uses Groq to generate a one-sentence explanation."""
    print(f"--- Groq: Explaining why '{rec_title}' is like '{original_title}' ---")
    prompt = ChatPromptTemplate.from_template(
    """
    You are a professional book analyst. Your job is to write a single, concise sentence 
    explaining what two books have in common, based on their descriptions.
    
    CRITICAL RULE: DO NOT be conversational. DO NOT ask questions. DO NOT include greetings.
    Your response MUST be ONLY the single explanatory sentence.
    
    Original Book: "{original_title}"
    Recommended Book: "{rec_title}"
    Recommended Book's Data: "{rec_doc}"
    
    Explain the similarity:
    """
)
    chain = prompt | groq_chat_model
    try:
        response = await chain.ainvoke({"original_title": original_title, "rec_title": rec_title, "rec_doc": rec_doc})
        return response.content
    except Exception as e:
        return "This book was found to be semantically similar."

async def format_final_response(query: str, recommendations: List[Recommendation]) -> str:
    """Uses Groq to turn the final list into a conversational response."""
    print("--- Groq: Formatting final conversational response ---")
    rec_list_str = "\n".join([f"- {rec.title}: {rec.explanation}" for rec in recommendations])
    prompt = ChatPromptTemplate.from_template("just repharse this: 'i think these might be best recommendation of books for you'")
    chain = prompt | groq_chat_model
    try:
        response = await chain.ainvoke({"query": query, "recommendations": rec_list_str})
        return response.content
    except Exception as e:
        return "Here are the recommendations I found for you:"

# === RESTORED: SYNCHRONOUS Summary Function (Stable) ===
def get_book_summary_sync(book_title: str) -> str:
    """
    Synchronous function for the summary agent (called via asyncio.to_thread).
    """
    print(f"--- Summary Agent: Starting for '{book_title}' (Sync) ---")
    agent_input = f"Get a detailed description and plot summary for the book/topic/series/or get books of the genere if you can find books of that name then fetch keywords that we most pervalant: '{book_title}' and give the final answer in one paragraph. (Preference must be given to finding an actual, available book with this title; only use the topic/hypothetical summary method if the title is unfindable.)"
    raw_output = ""
    
    try:
        print("--- Summary: Calling Gemini Agent ---")
        raw_text = summary_agent_executor.invoke({"input": agent_input})
        raw_output = raw_text.get('output', 'No information found.')

    except Exception as e:
        print(f"!!! Gemini Agent FAILED: {e}. Trying HF Backup... !!!")
        raw_output = call_hf_api_sync(agent_input)

    if not raw_output or raw_output == 'No information found.':
        return "I couldn't find any information about that book to summarize."

    print("--- Summary: Agent finished, returning result. ---")
    return raw_output