import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Local Embedding Model
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  
    
    # GPT-3.5 Turbo for answering questions
    GPT_MODEL = "gpt-3.5-turbo"
    
    # Chunking Settings
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 128
    TOP_K = 3  # Number of chunks to retrieve
    MAX_TOKENS = 500  # Max tokens for GPT response
    
    # Batch Processing
    BATCH_SIZE = 512  # Smaller batch size for memory efficiency
    
    @staticmethod
    def api_key():
        return os.getenv("OPENAI_API_KEY")