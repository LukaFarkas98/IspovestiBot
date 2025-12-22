import re
import unicodedata
import json

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



INPUT_FILE = "confessions_Archive_100k.jsonl"
OUTPUT_FILE = "confessions_Archive_100k_clean_TRAINING.jsonl"


PREFIX = "Ispovest:\n"

def clean_text(text: str) -> str:
    # Unicode normalize (fix weird chars)
    text = unicodedata.normalize("NFKC", text)

    # Remove non-breaking spaces & control chars
    text = text.replace("\xa0", " ")
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e]", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

    for line in fin:
        item = json.loads(line)

        text = clean_text(item["text"])
        if not text:
            continue

        item["text"] = PREFIX + text
        #fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        fout.write(json.dumps({"text": item["text"]}, ensure_ascii=False) + "\n")

print("✅ Cleaning done →", OUTPUT_FILE)
