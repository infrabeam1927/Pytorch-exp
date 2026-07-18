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
        documents = results['documents'][0] if results['documents'] else []
        metadatas = results['metadatas'][0] if results.get('metadatas') else [{}] * len(documents)
        return list(zip(documents, metadatas))

    def generate_answer(self, question, context):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return None

        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        prompt = (
            "Answer the question using only the context below. "
            "If the context doesn't contain the answer, say so.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}"
        )
        try:
            response = client.messages.create(
                model=DEFAULT_MODEL,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
        except anthropic.APIError as exc:
            print(f"\n(Answer generation failed ({exc}); showing retrieved context only.)")
            return None
        return response.content[0].text

    def ask(self, question):
        results = self.retrieve(question)

        if not results:
            print("No relevant context found in the database.")
            return None

        print("\n--- RELEVANT CONTEXT FOUND ---")
        for doc, meta in results:
            source = meta.get("source") or meta.get("plan") or "unknown source"
            print(f"[{source}] {doc[:200]}...")

        context = "\n\n".join(doc for doc, _ in results)
        answer = self.generate_answer(question, context)
        if answer:
            print("\n--- ANSWER ---")
            print(answer)
        else:
            print(
                "\n(Set ANTHROPIC_API_KEY to generate a synthesized answer "
                "from the retrieved context above.)"
            )

        return answer or [doc for doc, _ in results]


if __name__ == "__main__":
    try:
        pipeline = RAGPipeline()
    except RuntimeError as exc:
        print(exc)
        sys.exit(1)

    user_query = input("Ask a question about the PRT Onboarding process: ")
    pipeline.ask(user_query)
