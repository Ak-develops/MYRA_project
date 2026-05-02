# 🔐 Retrieval-based Teaching Assistant (RITA)

A Retrieval-Augmented Generation (RAG) system designed to answer NLP and LLM-related queries using trusted academic resources such as *Introduction to Natural Language Processing* by Tong Xiao and Jingbo Zhu.

Instead of relying purely on LLM memory (which is prone to hallucination), RITA grounds responses in specific academic documents to provide technically accurate and context-aware information.

---

## 🚀 Problem Statement

Traditional LLMs often:
- Hallucinate technical details regarding NLP architectures
- Provide generic advice that lacks academic depth
- Fail to reference specific textbook methodologies

### ✅ RITA solves this by:
- Retrieving relevant chunks from NLP and LLM textbooks
- Generating context-grounded answers based on retrieved data
- Reducing hallucination risk through strict grounding checks

---

## 🧠 System Flow

```
User Query 
   → FAISS Retriever 
   → Top-K Chunks 
   → Cross-Encoder Reranker 
   → Prompt Builder 
   → LLM (Gemini / Groq) 
   → Response
```

---

## 🏗️ Architecture

The system follows a **modular RAG architecture**, separating:
- Data ingestion
- Retrieval logic
- Generation pipeline

This makes the system scalable, testable, and easy to extend.

---

## ⚙️ Tech Stack

### LLMs
- Gemini (2.5 Flash / 3 Flash Preview)
- Groq (LLaMA 3.3 70B / 3.1 8B)

### Embeddings
- `all-MiniLM-L6-v2` (Sentence-Transformers)

### Vector Database
- FAISS (CPU)

### Backend
- Python 3.10 + LangChain

### Frontend
- Streamlit

---

## ✨ Features

- **PDF Ingestion Pipeline**  
  Automated extraction and cleaning of textbook PDFs using `PyPDFLoader`

- **Two-Stage Retrieval**
  - FAISS semantic search (recall)
  - Cross-Encoder reranker (precision)

- **Semantic Conversation Memory**  
  Retrieves relevant past interactions using embeddings

- **Multi-LLM Resilience**  
  Automatic fallback between Groq and Gemini if APIs fail

- **Grounding Evaluation**  
  Confidence scoring using cosine similarity between answer and retrieved context

---

## 📂 Project Structure

```text
rag_project/
│
├── app/                    # Streamlit UI
├── data/                   # Source PDFs
├── docs/                   # Architecture diagrams
├── evaluation/             # Grounding + confidence scripts
├── faiss_index/            # Vector DB storage
├── LLM_Prompts/            # Prompt templates
├── src/                    # Core logic
│   ├── ingest.py           # Document processing + indexing
│   ├── retriever.py        # Two-stage retrieval
│   ├── rag_core.py         # Pipeline orchestration
│   ├── llm.py              # LLM routing + fallback
│   ├── memory.py           # Conversation memory
│   └── prompt_builder.py   # Prompt construction
│
├── .env                    # API keys
├── requirements.txt        # Dependencies
└── README.md               # Documentation
```

---

## 🛡️ Academic Focus

- **Primary Source:** *Introduction to Natural Language Processing* (Tong Xiao & Jingbo Zhu)
- **Domain:** NLP and Large Language Models

This ensures:
- Alignment with academic standards
- Reduced hallucination in educational contexts

---

## 🧪 Evaluation

### Metric
- Weighted Cosine Similarity

### Process
Measures alignment between:
- User query
- Retrieved chunks
- Generated response

### Interpretation
- **Score > 0.75** → Highly grounded  
- **Score < 0.40** → Likely hallucinated → system responds with *"I don't know"*

---

## 🚀 Setup & Run

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd rag_project
```

### 2. Create Environment
```bash
conda create -n rita_env python=3.10
conda activate rita_env
pip install -r requirements.txt
```

### 3. Setup Environment Variables
Create a `.env` file in the root directory and add your API keys.

### 4. Run Application
```bash
streamlit run app/app.py
```

---

## ⚠️ Limitations & Future Improvements

### Current Limitations
- Memory system uses hybrid importance trimming + semantic retrieval

### Future Improvements
- Hybrid Search (BM25 + Dense retrieval)
- Integration with RAG evaluation frameworks (e.g., RAGAS)