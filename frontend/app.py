import streamlit as st
import PyPDF2
import docx
import requests

# =========================
# BACKEND ENDPOINTS
# =========================
UPLOAD_URL = "http://127.0.0.1:8000/upload-resumes"
MATCH_URL = "http://127.0.0.1:8000/match-text"

st.set_page_config(
    page_title="GenAI Resume Matcher",
    layout="wide"
)

st.title("🤖 GenAI Resume & Job Matching System")
st.write("Upload resumes, index them, and match against a job description.")

st.divider()

# =========================
# Layout
# =========================
left_col, right_col = st.columns(2)

# =========================
# LEFT: Job Description
# =========================
with left_col:
    st.subheader("📄 Job Description")

    jd_input_type = st.radio(
        "Choose JD input method",
        options=["Paste Text", "Upload PDF"]
    )

    job_text = ""

    if jd_input_type == "Paste Text":
        job_text = st.text_area(
            "Paste Job Description",
            height=250,
            placeholder="Enter job description here..."
        )
    else:
        jd_file = st.file_uploader("Upload Job Description PDF", type=["pdf"])

        if jd_file:
            reader = PyPDF2.PdfReader(jd_file)
            job_text = " ".join(
                page.extract_text() for page in reader.pages if page.extract_text()
            )
            st.success("Job Description loaded")

# =========================
# RIGHT: Resume Upload
# =========================
with right_col:
    st.subheader("📑 Upload Resumes")

    resume_files = st.file_uploader(
        "Upload multiple resumes (PDF or DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )

    if resume_files:
        st.success(f"{len(resume_files)} resumes selected")
st.divider()

# =========================
# MATCH BUTTON
# =========================
if st.button("🔍 Match Candidates", use_container_width=True):
    if not job_text.strip():
        st.warning("Please provide a Job Description.")
    else:
        with st.spinner("Matching candidates using FAISS + LLM..."):
            payload = {
                "text": job_text,
                "top_k": 5
            }

            try:
                response = requests.post(MATCH_URL, json=payload)

                if response.status_code != 200:
                    st.error("Backend error. Please check FastAPI server.")
                else:
                    data = response.json()

                    if "results" not in data or len(data["results"]) == 0:
                        st.warning("No matching candidates found.")
                    else:
                        st.success("Top matching candidates")

                        for idx, res in enumerate(data["results"], start=1):
                            st.subheader(f"#{idx} — {res['file_name']}")
                            st.write(f"**Predicted Role:** {res['predicted_role']}")
                            st.write(f"**Similarity Score:** {round(res['similarity_score'], 3)}")
                            st.markdown("**🧠 AI Explanation:**")
                            st.write(res["llm_explanation"])
                            st.markdown("---")

            except Exception as e:
                st.error(f"Error connecting to backend: {e}")
