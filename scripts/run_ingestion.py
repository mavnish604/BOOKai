# scripts/run_ingestion.py
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
import sys
from pathlib import Path

# --- Add Project Root to Python Path ---
# This allows us to import from the 'bookai' package
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))
# -------------------------------------

# Import paths from our new config file
from bookai.config import RAW_CSV_PATH, DB_PERSIST_PATH

# --- 1. Load Data and Create Documents ---
print(f"Loading data from {RAW_CSV_PATH}...")
df = pd.read_csv(RAW_CSV_PATH)
records = df.to_dict("records")

documents = []
for record in records:
    page_content = (
        f"Title: {record['Title']}. "
        f"Authors: {record['Authors']}. "
        f"Description: {record['Description']}. "
        f"Category: {record['Category']}"
    )
    metadata = {
        'category': record['Category'],
        'publisher': record['Publisher'],
        'source_title': record['Title']
    }
    doc = Document(page_content=page_content, metadata=metadata)
    documents.append(doc)

# --- 2. Initialize Embeddings and Persistent Vector Store ---
print("Initializing embedding model...")
embedding = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

print(f"Initializing vector store at {DB_PERSIST_PATH}...")
vector_store = Chroma(
    embedding_function=embedding,
    persist_directory=str(DB_PERSIST_PATH), # Use the config path (and cast to string)
    collection_name="BOOKAIDB"
)

# --- 3. Batch Insertion Function ---
MAX_BATCH_SIZE = 5000 

def batch_insert_documents(vector_store: Chroma, documents: list[Document], batch_size: int):
    """Inserts a large list of documents into the vector store in smaller batches."""
    total_documents = len(documents)
    print(f"Total documents to upload: {total_documents}")
    num_batches = (total_documents + batch_size - 1) // batch_size
    
    for i in range(0, total_documents, batch_size):
        batch = documents[i:i + batch_size]
        print(f"Uploading batch {i // batch_size + 1}/{num_batches}...")
        try:
            vector_store.add_documents(batch)
            print(f"Batch {i // batch_size + 1} successfully uploaded.")
        except Exception as e:
            print(f"CRITICAL ERROR during batch {i // batch_size + 1} upload: {e}")
            raise 

# --- 4. Execution ---
print("Starting batch document insertion...")
batch_insert_documents(vector_store, documents, MAX_BATCH_SIZE)
print("-" * 80)
print(f"All document batches processed. Data is persisted to '{DB_PERSIST_PATH}'.")