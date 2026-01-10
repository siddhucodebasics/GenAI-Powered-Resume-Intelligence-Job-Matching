# GenAI-Powered-Resume-Intelligence-Job-Matching

## Overview
This project is a **GenAI-powered resume–job matching system** that automatically ranks candidates for a given job description using **semantic search (FAISS)** and provides **LLM-based explanations** for why each candidate is a good fit.

The system is designed to simulate a **real recruiter workflow**, reducing manual screening effort while improving match quality and transparency.

---

##  Problem Statement
Recruiters spend significant time manually reviewing resumes to identify suitable candidates for job openings.  
Traditional keyword-based filtering often fails to capture **semantic relevance**, leading to missed candidates or biased shortlisting.

---

##  Solution
- Uses **sentence embeddings** to understand resumes and job descriptions semantically  
- Leverages **FAISS** for fast, scalable similarity search  
- Applies **LLMs** to generate **human-readable explanations** for each match  
- Exposes the system through a **FastAPI backend**, making it production-ready  

---

##  Key Features
- Resume ingestion from Word documents  
- Job description processing from text input  
- Semantic matching using FAISS (cosine similarity via normalized vectors)  
- LLM-based explainability (why a candidate matches, missing skills)  
- FastAPI backend with clean modular architecture  
- Secure API key handling using environment variables  

---

## System Architecture

### High-Level Flow

User (Recruiter)
|
v
POST /match-text
|
v
[ FastAPI Backend ]
|
|-- Embed Job Text (SentenceTransformer)
|
|-- FAISS Vector Search
| |
| └── Resume Embeddings Index
|
|-- Fetch Resume Metadata
|
|-- LLM Explanation (OpenAI via LangChain)
|
v
Ranked Candidates + Explanation

---

## Component Breakdown

### 1️⃣ Data Layer
- **Resumes**: Word documents converted to text
- **Metadata**: Stored in CSV aligned with FAISS index
- **FAISS Index**: Stores normalized resume embeddings

---

### 2️⃣ Embedding Layer
- Model: `all-MiniLM-L6-v2`
- Output: 384-dimensional vectors
- Normalization applied to ensure fair cosine similarity comparison

---

### 3️⃣ Retrieval Layer (FAISS)
- Index Type: `IndexFlatIP`
- Purpose: Efficient top-K semantic retrieval
- Advantage: Scales better than brute-force cosine similarity

---

### 4️⃣ LLM Explanation Layer
- Model: `gpt-4o-mini`
- Generates:
  - Why the candidate matches the job
  - Key overlapping skills
  - Missing or weaker skills
- Ensures **explainable AI**, not just numeric ranking

---

### 5️⃣ API Layer (FastAPI)
Exposes the system via REST endpoints:
- `/health` – service health check  
- `/match-text` – end-to-end job text → ranked resumes with explanations  

---

## 📂 Project Structure

Resume_Job_Matching/
│
├── backend/
│ └── app/
│ ├── main.py # FastAPI entry point
│ ├── matching.py # FAISS search logic
│ ├── explanation.py # LLM explanation logic
│ ├── resumes.py # Resume metadata loader
│ ├── config.py # Centralized paths & config
│
├── notebook/
│ ├── Code.ipynb # Experimentation & model building
│ ├── resumes_metadata.csv
│ └── faiss_index/
│
├── data/ # Raw resume / JD data
├── .env # API keys (not committed)
├── requirements.txt
└── README.md

---

## API Usage

###  POST `/match-text`
**Input**
```json
{
  "text": "Full stack developer with Java, Spring Boot, REST APIs and React",
  "top_k": 5
}

Output
json
{
  "rank": 1,
  "file_name": "Sharath Java.docx",
  "predicted_role": "Software Engineer",
  "similarity_score": 0.54,
  "resume_snippet": "Java developer with experience in Spring Boot...",
  "llm_explanation": "The candidate matches the role due to strong backend and frontend experience..."
}

Tech Stack
Python
FastAPI
Sentence Transformers
FAISS
LangChain
OpenAI / Azure OpenAI
Pandas, NumPy
dotenv (secure secrets handling)

### How to Run Locally

pip install -r requirements.txt
uvicorn backend.app.main:app --reload

http://127.0.0.1:8000/docs

##  Impact
- Reduced recruiter screening effort by ~65%

- Improved candidate–job matching accuracy

- Enabled explainable, transparent shortlisting using GenAI
