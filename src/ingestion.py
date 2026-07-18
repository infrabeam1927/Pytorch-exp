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

    def extract_and_chunk(self, pdf_path, chunk_size=500, overlap=50):
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        reader = PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            full_text += (page.extract_text() or "") + " "

        if not full_text.strip():
            raise ValueError(f"No extractable text found in {pdf_path}")

        return self._chunk_words(full_text.split(), chunk_size, overlap)

    def _chunk_words(self, words, chunk_size, overlap):
        # Word-aware chunking with a trailing overlap so context isn't
        # lost at chunk boundaries.
        chunks = []
        current = []
        current_len = 0
        for word in words:
            extra = len(word) + (1 if current else 0)
            if current and current_len + extra > chunk_size:
                chunks.append(" ".join(current))
                current, current_len = self._trailing_overlap(current, overlap)
                extra = len(word) + (1 if current else 0)
            current.append(word)
            current_len += extra
        if current:
            chunks.append(" ".join(current))
        return chunks

    def _trailing_overlap(self, words, overlap):
        tail = []
        tail_len = 0
        for word in reversed(words):
            extra = len(word) + (1 if tail else 0)
            if tail_len + extra > overlap:
                break
            tail.insert(0, word)
            tail_len += extra
        return tail, tail_len

    def run(self, pdf_path, plan_name=None):
        plan_name = plan_name or os.path.splitext(os.path.basename(pdf_path))[0]
        print(f"--- Processing Plan: {plan_name} ---")
        chunks = self.extract_and_chunk(pdf_path)

        # Metadata lets you filter searches by plan later
        metadatas = [{"plan": plan_name, "source": pdf_path} for _ in chunks]
        embeddings = self.model.encode(chunks).tolist()
        ids = [f"{plan_name}_{i}" for i in range(len(chunks))]

        # upsert (not add) so re-ingesting the same plan overwrites its
        # chunks instead of erroring on duplicate ids
        self.collection.upsert(
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