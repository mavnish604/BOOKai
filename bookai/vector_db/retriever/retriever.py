# bookai/vector_db/retriever/retriever.py
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# We import the *dynamic* path, not the hardcoded one
from bookai.config import DB_PERSIST_PATH 

embedding = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

# We use the config variable (and cast it to a string for Chroma)
vector_store = Chroma(
    collection_name="BOOKAIDB",
    embedding_function=embedding,
    persist_directory=str(DB_PERSIST_PATH) 
)

# --- TEST CODE AT THE END ---
if __name__ == "__main__":
    
    # --- THIS IS THE FIX ---
    # We must re-import config here for the __main__ scope
    from bookai.config import DB_PERSIST_PATH
    # -----------------------

    print("\n--- RETRIEVER TEST ---")
    print(f"Loading from: {DB_PERSIST_PATH}")
    print("Getting 5 documents from the DB...")
    
    # Get 5 documents to see if there is anything inside
    results = vector_store.get(limit=5, include=["metadatas"])
    
    if results and results.get('metadatas'):
        print(f"Found {len(results['metadatas'])} documents. The DB is populated.")
        for i, meta in enumerate(results['metadatas']):
            print(f"  {i+1}: {meta.get('source_title', 'NO TITLE')}")
    else:
        print("!!! ERROR: The database returned 0 documents. It is empty. !!!")
        print("!!! FIX: You need to run 'scripts/run_ingestion.py' to build the database. !!!")
    
    print("--- TEST COMPLETE ---")