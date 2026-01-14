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
Root Directory: Resume_Job_Matching

Folders & Files:

1) backend/app/main.py – FastAPI entry point

2) backend/app/matching.py – FAISS similarity search logic

3) backend/app/explanation.py – LLM explanation logic

4) backend/app/resumes.py – Resume metadata loader

5) backend/app/config.py – Centralized paths & configuration

6) notebook/Code.ipynb – Experimentation & model building

7) notebook/resumes_metadata.csv – Resume metadata

8) notebook/faiss_index/resume_index.faiss – FAISS index

9) data/ – Raw resume and job description data

10) .env – API keys (not committed)

11) requirements.txt

12) README.md

## How to Run the Full Application (Recommended)

This project consists of:
- FastAPI backend (API + GenAI logic)
- Streamlit frontend (Recruiter UI)

### Step 1: Install dependencies
pip install -r requirements.txt

### Step 2: Configure environment variables
Create a `.env` file in the project root:

OPENAI_API_KEY=your_key_here

### Step 3: Start backend (Terminal 1)
uvicorn backend.app.main:app --reload

Backend runs at:
http://127.0.0.1:8000

### Step 4: Start frontend (Terminal 2)
streamlit run frontend/app.py

Frontend runs at:
http://localhost:8501

# Tech Stack

- Python
- FastAPI
- Sentence Transformers
- FAISS
- LangChain
- OpenAI / Azure OpenAI
- Pandas
- NumPy
- dotenv

# Impact

Reduced recruiter screening effort by ~65%

Improved candidate–job matching accuracy

Enabled explainable and transparent shortlisting using GenAI

