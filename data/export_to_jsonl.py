import json
from database import SessionLocal
from models import Confession
from clean_utils import clean_text

OUTPUT_FILE = "confessions.jsonl"

def export():
    db = SessionLocal()

    confessions = db.query(Confession).all()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for c in confessions:
            record = {
                "text": clean_text(c.text)
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    db.close()
    print(f"Exported {len(confessions)} confessions to {OUTPUT_FILE}")




if __name__ == "__main__":
    export()
