# Pytorch-exp

A small local RAG (Retrieval-Augmented Generation) pipeline for PDF documents. It extracts text from a PDF, chunks it, embeds the chunks with a PyTorch-based sentence-transformer model, and stores them in a local ChromaDB vector store. You can then ask questions and get answers grounded in the retrieved chunks.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Optionally, to have answers synthesized by Claude instead of just returning raw retrieved chunks:

```bash
export ANTHROPIC_API_KEY=your-key-here
```

## Usage

1. Put a PDF in the `data/` folder (this folder is gitignored, so files placed there won't be committed).

2. Ingest the PDF into the vector store:

   ```bash
   python src/ingestion.py data/your-file.pdf
   ```

3. Ask questions against the ingested content:

   ```bash
   python src/main.py
   ```

   Without `ANTHROPIC_API_KEY` set, this prints the top matching chunks. With it set, it also prints an answer generated from that context.
