from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import pandas as pd
from langchain.schema import Document
# No need to import os if you rely on Chroma to create the directory

# --- 1. Load Data and Create Documents ---

# Ensure all columns are read correctly
df = pd.read_csv("/run/media/tst_imperial/Projects/BOOKai/VECTORDB/ProcessedDF.csv")
records = df.to_dict("records")

documents = []
for record in records:
    # 1. Augment and define the Content (page_content)
    # Combining Title, Authors, Description, and Category for rich embedding
    page_content = (
        f"Title: {record['Title']}. "
        f"Authors: {record['Authors']}. "
        f"Description: {record['Description']}. "
        f"Category: {record['Category']}" # Corrected f-string syntax
    )

    # 2. Define the Metadata (for filtering)
    metadata = {
        'category': record['Category'],
        'publisher': record['Publisher'],
        'source_title': record['Title']
    }
    doc = Document(page_content=page_content, metadata=metadata)
    documents.append(doc)

# --- 2. Initialize Embeddings and Persistent Vector Store ---

embedding = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

# Persistence is set here. Chroma will create the 'BOOKDB' directory
# and save data automatically upon object destruction (script end).
vector_store = Chroma(
    embedding_function=embedding,
    persist_directory="BOOKDB", 
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
        # Get the current batch slice
        batch = documents[i:i + batch_size]
        
        print(f"Uploading batch {i // batch_size + 1}/{num_batches}: "
              f"Documents {i + 1} to {min(i + batch_size, total_documents)}...")
        
        try:
            # Insert the smaller batch
            vector_store.add_documents(batch)
            print(f"Batch {i // batch_size + 1} successfully uploaded.")
        except Exception as e:
            # Added a critical error stop in case of a problem
            print(f"CRITICAL ERROR during batch {i // batch_size + 1} upload: {e}")
            raise # Stop execution to inspect the issue

# --- 4. Execution (One single call) ---

batch_insert_documents(vector_store, documents, MAX_BATCH_SIZE)


print(" All document batches have been processed. Data is automatically persisted to the 'BOOKDB' directory.")