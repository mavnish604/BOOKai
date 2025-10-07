from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

vector_store=Chroma(
    collection_name="BOOKAIDB",
    embedding_function=embedding,
    persist_directory="/run/media/tst_imperial/Projects/BOOKai/VECTORDB/BOOKDB"
)

result=vector_store.similarity_search_with_relevance_scores(
    query="Germs : Biological Weapons and America's Secre... 	By Miller, Judith, Engelberg, Stephen, and Bro... 	Deadly germs sprayed in shopping malls, bomb-l... 	Technology & Engineering , Military Science 	Simon & Schuster 	4.99 	October 	2001",
    k=3
)
for doc, score in result:
    # 1. Access the 'source_title' key from the Document's metadata dictionary
    title = doc.metadata.get('source_title', 'TITLE NOT FOUND')
    
    # 2. Print the formatted output
    # Use slicing [:] to ensure the title doesn't overflow if it's too long
    print("{:<70} | {:<10.4f}".format(title[:70], score)) 

print("-" * 82)