from dotenv import load_dotenv
load_dotenv()
from fastapi import UploadFile, File
from typing import List
import docx
import PyPDF2
import faiss
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from backend.app.resumes import load_resumes_csv
from pathlib import Path
from sentence_transformers import SentenceTransformer
from backend.app.config import FAISS_INDEX_PATH, RESUME_METADATA_PATH
from backend.app.matching import load_faiss_index, search_candidates
from backend.app.llm import generate_explanation

app = FastAPI(title="Resume Job Matching API")

# Load FAISS index when app starts
faiss_index = load_faiss_index(str(FAISS_INDEX_PATH))
RESUME_METADATA_PATH = Path("notebook/resumes_metadata.csv")
resumes_df = load_resumes_csv(RESUME_METADATA_PATH)

# Load embedding model once at startup
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# --------- Request Schema ---------
class MatchRequest(BaseModel):
    job_embedding: list[float]
    top_k: int = 5

# --------- Request Schema ---------
class EmbedRequest(BaseModel):
    text: str
# --------- Request Schema ---------
class MatchTextRequest(BaseModel):
    text: str
    top_k: int = 5

# --------- Health Check ---------
@app.get("/health")
def health_check():
    return {
        "status": "Backend is running",
        "faiss_vectors": faiss_index.ntotal
    }

# --------- Match Endpoint ---------
@app.post("/match")
def match_candidates(request: MatchRequest):
    """
    Match a job description embedding against resume embeddings using FAISS
    """
    job_embedding = np.array(request.job_embedding, dtype="float32")

    scores, indices = search_candidates(
        index=faiss_index,
        query_embedding=job_embedding,
        top_k=request.top_k
    )

    results = []
    for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
        results.append({
            "rank": rank,
            "resume_index": int(idx),
            "similarity_score": float(score)
        })

    return {
        "top_k": request.top_k,
        "results": results
    }

@app.post("/embed")
def embed_text(request: EmbedRequest):
    """
    Convert raw text into embedding
    """
    embedding = embedding_model.encode(request.text)
    return {
        "embedding": embedding.tolist(),
        "dimension": len(embedding)
    }

@app.post("/match-text")
def match_from_text(request: MatchTextRequest):
    try:
        # 1. Embed job text
        job_embedding = embedding_model.encode(request.text)

        # 2. FAISS search
        scores, indices = search_candidates(
            index=faiss_index,
            query_embedding=job_embedding,
            top_k=request.top_k
        )

        results = []

        for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
            resume = resumes_df.iloc[int(idx)]

            llm = generate_explanation(
                job_text=request.text,
                resume_text=resume["text"]
            )

            results.append({
                "rank": rank,
                "file_name": str(resume["file_name"]),
                "predicted_role": str(resume["predicted_role"]),
                "similarity_score": float(score),
                "resume_snippet": resume["text"][:300],
                "llm_explanation": llm
            })

        return {
            "top_k": request.top_k,
            "results": results
        }

    except Exception as e:
        return {"error": str(e)}

@app.post("/upload-resumes")
async def upload_resumes(files: List[UploadFile] = File(...)):
    new_texts = []
    new_filenames = []

    for file in files:
        text = ""

        if file.filename.endswith(".pdf"):
            reader = PyPDF2.PdfReader(file.file)
            text = " ".join(
                page.extract_text() for page in reader.pages if page.extract_text()
            )

        elif file.filename.endswith(".docx"):
            doc = docx.Document(file.file)
            text = " ".join(
                para.text for para in doc.paragraphs if para.text.strip()
            )

        if text.strip():
            new_texts.append(text)
            new_filenames.append(file.filename)

    if not new_texts:
        return {"message": "No valid resumes found."}

    # 🔹 Embed
    embeddings = embedding_model.encode(new_texts)
    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)

    # 🔹 Add to FAISS
    faiss_index.add(embeddings)

    # 🔹 Update metadata
    new_rows = []
    for fname, txt in zip(new_filenames, new_texts):
        new_rows.append({
            "file_name": fname,
            "text": txt,
            "predicted_role": "Uploaded"
        })

    global resumes_df
    resumes_df = pd.concat([resumes_df, pd.DataFrame(new_rows)], ignore_index=True)

    # 🔹 Persist
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    resumes_df.to_csv(RESUME_METADATA_PATH, index=False)

    return {
        "message": f"{len(new_rows)} resumes uploaded and indexed successfully."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)