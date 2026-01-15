import json
from database import engine, SessionLocal
from models import Confession

# create tables
# creates the database out of json data scrapped from Ispovesti.com
Confession.metadata.create_all(bind=engine)

approveTag = "approve"
disapproveTag = "disapprove"
confession_100k_path = "../Scraping/confessions_Archive_200k.jsonl"
def parse_int(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0

def ingest_json(path: str):
    db = SessionLocal()
    count = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)

            confession = Confession(
                text=item.get("text", "").strip(),
                approve_count=parse_int(item.get(approveTag)),
                disapprove_count=parse_int(item.get(disapproveTag)),
                timestamp_raw=item.get("timestamp"),
            )

            db.add(confession)
            count += 1

            # commit in batches to avoid memory blowup
            if count % 1000 == 0:
                db.commit()
                print(f"Inserted {count} confessions...")

    db.commit()
    db.close()
    print(f"Inserted {count} confessions total")


if __name__ == "__main__":
    ingest_json(confession_100k_path)
