import json

# ---------------- FILES ----------------
TOPIC_JSONL = "confessions_clusters_TOPICS_FIXED.jsonl"
ENGAGEMENT_JSON = "confessions_with_votes.json"
OUTPUT_JSONL = "confessions_FINAL_TRAINING.jsonl"

# ---------------- HELPERS ----------------
def normalize_text(text: str) -> str:
    text = text.replace("Ispovest:\n", "")
    return text.strip().lower()

def engagement_score(approve, disapprove):
    a = int(approve)
    d = int(disapprove)
    return (a - d) / (a + d + 1)

# ---------------- LOAD ENGAGEMENT DATA ----------------
text_to_engagement = {}

with open(ENGAGEMENT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

for obj in data:
    raw_text = obj.get("text", "")
    key = normalize_text(raw_text)

    score = engagement_score(
        obj.get("approve", 0),
        obj.get("disapprove", 0)
    )

    text_to_engagement[key] = score

print(f"Loaded engagement for {len(text_to_engagement)} confessions")

# ---------------- MERGE INTO TOPIC JSONL ----------------
matched = 0
missing = 0

with open(TOPIC_JSONL, "r", encoding="utf-8") as fin, \
     open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:

    for line_num, line in enumerate(fin, 1):
        obj = json.loads(line)

        raw_text = obj.get("text", "")
        key = normalize_text(raw_text)

        if key in text_to_engagement:
            obj["engagement_score"] = round(text_to_engagement[key], 4)
            matched += 1
        else:
            obj["engagement_score"] = 0.0
            missing += 1

        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ---------------- SUMMARY ----------------
print("✅ DONE")
print(f"✔ Engagement matched : {matched}")
print(f"⚠ Missing matches    : {missing}")
print(f"📄 Output file       : {OUTPUT_JSONL}")
