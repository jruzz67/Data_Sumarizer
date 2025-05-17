import os
from pathlib import Path
import fitz  # PyMuPDF
import pandas as pd

def extract_text_from_file(file_path: str) -> str:
    """
    Extract text from PDF, TXT, or Excel files.
    Returns the extracted text as a string.
    """
    file_path = Path(file_path)
    file_ext = file_path.suffix.lower()

    try:
        if file_ext == ".pdf":
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                page_text = page.get_text("text")
                if page_text:
                    text += page_text + "\n"
            doc.close()
            return text.strip() if text else ""

        elif file_ext in [".txt"]:
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read().strip()

        elif file_ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)
            # Convert dataframe to string, handle edge cases
            if df.empty:
                return ""
            # Ensure the result is a string, even if the dataframe contains a single number
            text = df.to_string(index=False)
            return str(text).strip()

        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
    except Exception as e:
        raise Exception(f"Error extracting text: {str(e)}")