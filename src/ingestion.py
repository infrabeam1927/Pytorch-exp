import os
import sys
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
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        reader = PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            full_text += (page.extract_text() or "") + " "

        if not full_text.strip():
            raise ValueError(f"No extractable text found in {pdf_path}")

        # Word-aware chunking so chunks don't cut words in half
        words = full_text.split()
        chunks = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if len(candidate) > chunk_size and current:
                chunks.append(current)
                current = word
            else:
                current = candidate
        if current:
            chunks.append(current)

        return chunks

    def run(self, pdf_path, plan_name=None):
        plan_name = plan_name or os.path.splitext(os.path.basename(pdf_path))[0]
        print(f"--- Processing Plan: {plan_name} ---")
        chunks = self.extract_and_chunk(pdf_path)

        # Metadata lets you filter searches by plan later
        metadatas = [{"plan": plan_name, "source": pdf_path} for _ in chunks]
        embeddings = self.model.encode(chunks).tolist()
        ids = [f"{plan_name}_{i}" for i in range(len(chunks))]

        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Stored {len(chunks)} chunks for {plan_name}.")


if __name__ == "__main__":
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not pdf_path:
        print("Usage: python src/ingestion.py <path-to-pdf>")
        sys.exit(1)

    pipeline = IngestionPipeline()
    pipeline.run(pdf_path)