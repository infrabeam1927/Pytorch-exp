import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

class IngestionPipeline:
    def __init__(self, db_path="./chroma_db"):
        # 1. Initialize the Embedding Model (PyTorch-based)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 2. Initialize Persistent ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name="pdf_documents")

    def extract_and_chunk(self, pdf_path, chunk_size=500):
        reader = PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + " "
        
        # Simple character-based chunking
        chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
        return chunks

    def run(self, pdf_path):
        print(f"--- Processing: {pdf_path} ---")
        chunks = self.extract_and_chunk(pdf_path)
        
        # 3. Create Embeddings and Store
        embeddings = self.model.encode(chunks).tolist()
        ids = [f"{os.path.basename(pdf_path)}_{i}" for i in range(len(chunks))]
        
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids
        )
        print(f"Successfully stored {len(chunks)} chunks in the vector database.")
def run(self, pdf_path, plan_name):
        print(f"--- Processing Plan: {plan_name} ---")
        chunks = self.extract_and_chunk(pdf_path)
        
        # Adding metadata lets you filter searches by plan later
        metadatas = [{"plan": plan_name, "source": pdf_path} for _ in chunks]
        embeddings = self.model.encode(chunks).tolist()
        ids = [f"{plan_name}_{i}" for i in range(len(chunks))]
        
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas, # <--- NEW: Professional data handling
            ids=ids
        )
        print(f"Stored {len(chunks)} chunks for {plan_name}.")

if __name__ == "__main__":
    pipeline = IngestionPipeline()
    # Replace with a path to a real PDF in your 'data/' folder
    pipeline.run("data/Associate PRT Onboarding and Client Experience - Blumont Annuity.pdf")