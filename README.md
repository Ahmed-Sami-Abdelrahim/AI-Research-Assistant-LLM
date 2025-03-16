# **AI Research Assistant LLM**  

AI Research Assistant LLM is a **Retrieval-Augmented Generation (RAG)** system designed to help researchers quickly find answers using **academic papers from ArXiv**. It utilizes **FAISS indexing** for fast retrieval and **GPT-based text generation** for accurate responses.

---

## **Features**  
- ✅ **FAISS-powered search** for efficient retrieval  
- ✅ **GPT-generated answers** using relevant research papers  
- ✅ **Streamlit UI** for easy interaction  
- ✅ **Embeddings with SentenceTransformer** (`all-MiniLM-L6-v2`)  

---

## **Project Structure & Notes**  

```plaintext
AI-Research-Assistant-LLM/
│── data/                   # Storage for datasets and processed information
│   ├── raw/                # Raw data (e.g., ArXiv metadata)
│   ├── processed/          # Processed chunks, embeddings, and FAISS index
│── build_index.py          # Creates FAISS index from processed data
│── answer_generator.py     # Retrieves context & generates GPT responses
│── app.py                  # Streamlit UI for research queries
│── config.py               # Configuration settings
│── requirements.txt        # Dependencies
│── README.md               # Documentation
│── .env                    # API keys and environment variables (not included)


## 📌 Notes  

### 1️⃣ Dataset Availability  
The dataset (`arxiv-metadata-oai-snapshot.json`) is **NOT included** due to size constraints.  

You need to **download the dataset manually** and place it in `data/raw/`.  
Without this dataset, the **FAISS index cannot be built**.  

### 2️⃣ API Keys  
The project requires an **API key** for GPT-based text generation.  

- Store your **OpenAI API key** in a `.env` file.  
- Format:  
  ```env
  OPENAI_API_KEY=your_api_key_here
