import re
import unicodedata

def clean_text(text: str) -> str:
    if not text:
        return ""
    
    # Normalize Unicode (NFKC removes weird forms)
    text = unicodedata.normalize("NFKC", text)
    
    # Replace newlines and tabs with space
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    
    # Replace en-dash / em-dash with normal dash
    text = text.replace("–", "-").replace("—", "-")
    
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)
    
    # Strip leading/trailing spaces
    text = text.strip()
    
    return text
