# bookai/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# 1. Load environment variables from .env file
load_dotenv()

# 2. Define the Project's Base Directory
# This finds the 'BOOKai' root folder (where .env is)
PROJECT_ROOT = Path(__file__).parent.parent

# 3. Define all critical paths relative to the PROJECT_ROOT
DATA_DIR = PROJECT_ROOT / "Data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "test"

# 4. Define specific file paths
RAW_CSV_PATH = RAW_DATA_DIR / "ProcessedDF.csv"
BOOKS_PKL_PATH = PROCESSED_DATA_DIR / "books.pkl"
SIMI_MATRIX_PATH = PROCESSED_DATA_DIR / "simi_sparse.npz"

# 5. Define Vector DB Path
# This is the path Chroma will use for persistence
DB_PERSIST_PATH = VECTOR_STORE_DIR / "BOOKDB"

# 6. API Keys
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# bookai/config.py

# 1. Load environment variables from .env file
load_dotenv()

# --- (Your existing path variables) ---
PROJECT_ROOT = Path(__file__).parent.parent
# ... (all your other paths) ...
DB_PERSIST_PATH = VECTOR_STORE_DIR / "BOOKDB"

# --- 6. API Keys ---
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # We'll need this for Gemini

# --- ADD THIS LINE ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")