import pandas as pd
from pathlib import Path

def load_resumes_csv(path: Path) -> pd.DataFrame:
    """
    Load resume metadata (index-aligned with FAISS)
    """
    df = pd.read_csv(path)
    return df