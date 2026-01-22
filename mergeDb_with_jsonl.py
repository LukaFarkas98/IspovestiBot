import os
import json
import re
import unicodedata
from sqlalchemy.orm import Session
from data.models import Confession  # your ORM model
from data.database import Base, engine  # your SQLAlchemy Base and engine

# ---------------- FILE PATHS ----------------
TOPIC_JSONL = "confessions_clusters_TOPICS_FIXED_NEED_SCORE.jsonl"  # input JSONL
OUTPUT_JSONL = "confessions_FINAL_WITH_ENGAGEMENT.jsonl"  # output JSONL

# ---------------- HELPERS ----------------
def normalize_text(text: str) -> str:
    """
    Normalize text for matching:
    - remove 'Ispovest:\n' prefix
    - lowercase
    - collapse multiple spaces
    - strip
    - normalize unicode accents
    """
    text = text.replace("Ispovest:\n", "")
    text = text.lower()
    text = re.sub(r"\s+", " ", text.strip())
    text = unicodedata.normalize("NFC", text)
    return text

def engagement_score(approve, disapprove) -> float:
    """Compute normalized engagement score in [-1,1]"""
    a = int(approve)
    d = int(disapprove)
    return round((a - d) / (a + d + 1), 4)

# ---------------- ENSURE TABLE EXISTS ----------------
Base.metadata.create_all(bind=engine)

# ---------------- LOAD DATABASE ----------------
print("Loading engagement data from database...")

db = Session(bind=engine)
text_to_engagement = {}

confessions = db.query(
    Confession.text,
    Confession.approve_count,
    Confession.disapprove_count
).all()

for conf_text, approve, disapprove in confessions:
    key = normalize_text(conf_text)  # normalize DB text
    score = engagement_score(approve, disapprove)
    text_to_engagement[key] = score

db.close()
print(f"Loaded engagement for {len(text_to_engagement)} confessions")

# ---------------- MERGE WITH JSONL ----------------
matched = 0
missing = 0

with open(TOPIC_JSONL, "r", encoding="utf-8") as fin, \
     open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:

    for line_num, line in enumerate(fin, 1):
        obj = json.loads(line)
        key = normalize_text(obj.get("text", ""))  # normalize JSONL text

        if key in text_to_engagement:
            obj["engagement_score"] = text_to_engagement[key]
            matched += 1
        else:
            obj["engagement_score"] = 0.0
            missing += 1

        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ---------------- SUMMARY ----------------
print("✅ DONE")
print(f"✔ Matched: {matched}")
print(f"⚠ Missing: {missing}")
print(f"📄 Output file: {OUTPUT_JSONL}")
