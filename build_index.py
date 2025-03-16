import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config import Config

# Load the embedding model
embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)

def chunk_text(text, chunk_size=Config.CHUNK_SIZE, overlap=Config.CHUNK_OVERLAP):
    """Splits text into overlapping chunks for retrieval."""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i : i + chunk_size])
    return chunks

def process_data():
    """Loads the dataset, chunks text, generates embeddings, and saves to FAISS index."""
    with open("data/raw/arxiv-metadata-oai-snapshot.json", "r") as f:
        papers = [json.loads(line) for line in f]

    # Store processed text chunks
    all_chunks = []
    all_embeddings = []

    for paper in papers:
        text_chunks = chunk_text(paper.get("abstract", ""))
        all_chunks.extend([{"id": len(all_chunks), "text": chunk} for chunk in text_chunks])

        # Generate embeddings
        embeddings = embedding_model.encode(text_chunks)
        all_embeddings.append(embeddings)

    # Convert to FAISS index
    embedding_matrix = np.vstack(all_embeddings).astype("float32")
    index = faiss.IndexFlatL2(embedding_matrix.shape[1])
    index.add(embedding_matrix)

    # Save processed chunks and FAISS index
    with open("data/processed/chunks.json", "w") as f:
        json.dump(all_chunks, f)

    faiss.write_index(index, "data/processed/faiss_index.bin")

    print("FAISS index built successfully!")

if __name__ == "__main__":
    process_data()
