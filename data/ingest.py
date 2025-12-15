import json
from database import engine, SessionLocal
from models import Confession

# create tables
# creates the database out of json data scrapped from Ispovesti.com
Confession.metadata.create_all(bind=engine)

def parse_int(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0

def ingest_json(path: str):
    db = SessionLocal()

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        confession = Confession(
            text=item.get("text", "").strip(),
            approve_count=parse_int(item.get("approve_count")),
            disapprove_count=parse_int(item.get("disapprove_count")),
            timestamp_raw=item.get("timestamp"),
        )

        db.add(confession)

    db.commit()
    db.close()
    print(f"Inserted {len(data)} confessions")

if __name__ == "__main__":
    ingest_json("confessions_page.json")
