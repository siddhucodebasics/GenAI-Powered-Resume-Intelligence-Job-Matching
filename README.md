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
Request Flow (Step-by-Step):

1) Recruiter submits a job description via POST /match-text

2) FastAPI backend receives the request

3) Job text is converted into embeddings using SentenceTransformer

4) FAISS performs semantic vector search over resume embeddings

5) Resume metadata is fetched using FAISS indices

6) LLM (OpenAI via LangChain) generates explanation for each match

7) API returns ranked candidates with explanations

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
├── backend/
│   └── app/
│       ├── main.py            # FastAPI entry point
│       ├── matching.py        # FAISS similarity search logic
│       ├── explanation.py     # LLM explanation logic
│       ├── resumes.py         # Resume metadata loader
│       └── config.py          # Centralized paths & configuration
│
├── notebook/
│   ├── Code.ipynb             # Experimentation & model building
│   ├── resumes_metadata.csv
│   └── faiss_index/
│
├── data/
│   └── raw_resumes_and_jds/
│
├── .env                       # API keys (not committed)
├── requirements.txt
└── README.md

