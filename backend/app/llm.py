from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Load LLM once
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    max_tokens = 200
)

prompt_template = PromptTemplate(
    input_variables=["job_text", "resume_text"],
    template="""
You are an AI assistant helping recruiters.

Job Description:
{job_text}

Candidate Resume:
{resume_text}

Explain clearly:
1. Why this candidate matches the job in 30 words
2. Key overlapping skills in 30 words
3. Missing or weaker skills (if any) in 30 words

Be concise and factual. Do not hallucinate.
"""
)

def generate_explanation(job_text: str, resume_text: str) -> str:
    prompt = prompt_template.format(
        job_text=job_text,
        resume_text=resume_text[:1500]
    )
    response = llm.invoke(prompt)
    return response.content