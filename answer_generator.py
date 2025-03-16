import json
import openai
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config import Config

# Load AI models and FAISS index
client = openai.OpenAI(api_key=Config.api_key())
index = faiss.read_index("data/processed/faiss_index.bin")
with open("data/processed/chunks.json", "r") as f:
    chunks = json.load(f)
embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)

def retrieve_context(query):
    """Retrieve relevant context using FAISS."""
    query_embedding = embedding_model.encode([query]).astype("float32")
    _, indices = index.search(query_embedding, Config.TOP_K)
    return [chunks[i] for i in indices[0]]

def generate_answer(question):
    """Generate an AI-powered response using GPT-3.5 Turbo."""
    context = retrieve_context(question)
    if not context:
        return "No relevant context found."

    system_msg = "You are a research assistant. Use the provided context to answer questions."
    user_msg = f"Context: {[c['text'] for c in context]}\n\nQuestion: {question}"

    try:
        response = client.chat.completions.create(
            model=Config.GPT_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=Config.MAX_TOKENS
        )
        return response.choices[0].message.content
    except openai.RateLimitError:
        return "API rate limit exceeded - please try again later."
    except Exception as e:
        return f"Error generating answer: {str(e)}"
