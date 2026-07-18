import os
import sys

import chromadb
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = os.environ.get("ANSWER_MODEL", "claude-haiku-4-5-20251001")


class RAGPipeline:
    def __init__(self, db_path="./chroma_db", collection_name="pdf_documents"):
        self.client = chromadb.PersistentClient(path=db_path)
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except Exception as exc:
            raise RuntimeError(
                f"Collection '{collection_name}' not found at {db_path}. "
                "Run src/ingestion.py first to ingest a PDF."
            ) from exc
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    def retrieve(self, question, n_results=2):
        query_vec = self.embed_model.encode([question]).tolist()
        results = self.collection.query(
            query_embeddings=query_vec,
            n_results=n_results
        )
        return results['documents'][0] if results['documents'] else []

    def generate_answer(self, question, context_chunks):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return None

        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        context = "\n\n".join(context_chunks)
        prompt = (
            "Answer the question using only the context below. "
            "If the context doesn't contain the answer, say so.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}"
        )
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    def ask(self, question):
        context_chunks = self.retrieve(question)

        if not context_chunks:
            print("No relevant context found in the database.")
            return None

        print("\n--- RELEVANT CONTEXT FOUND ---")
        for doc in context_chunks:
            print(f"- {doc[:200]}...")

        answer = self.generate_answer(question, context_chunks)
        if answer:
            print("\n--- ANSWER ---")
            print(answer)
        else:
            print(
                "\n(Set ANTHROPIC_API_KEY to generate a synthesized answer "
                "from the retrieved context above.)"
            )

        return answer or context_chunks


if __name__ == "__main__":
    try:
        pipeline = RAGPipeline()
    except RuntimeError as exc:
        print(exc)
        sys.exit(1)

    user_query = input("Ask a question about the PRT Onboarding process: ")
    pipeline.ask(user_query)
