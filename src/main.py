import chromadb
from sentence_transformers import SentenceTransformer

# 1. Setup - Connect to the "Brain" you built in Ingestion
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="pdf_documents")
model = SentenceTransformer('all-MiniLM-L6-v2')

def ask_question(question):
    # 2. Convert question to numbers (Vector)
    query_vec = model.encode([question]).tolist()
    
    # 3. Search the DB for the top 2 most relevant chunks
    results = collection.query(
        query_embeddings=query_vec,
        n_results=2
    )
    
    # 4. Display the results
    print("\n--- RELEVANT CONTEXT FOUND ---")
    for doc in results['documents'][0]:
        print(f"- {doc[:200]}...") # Print first 200 chars
    
    return results['documents'][0]

if __name__ == "__main__":
    user_query = input("Ask a question about the PRT Onboarding process: ")
    ask_question(user_query)