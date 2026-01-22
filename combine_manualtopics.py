import json
import re

# ---------------- FILES ----------------
INPUT_JSONL = "confessions_clusters_INCLUDE_MORE_DATA.jsonl"
MANUAL_TOPICS_TXT = "topics_manual_extracted.txt"
OUTPUT_JSONL = "confessions_clusters_TOPICS_FIXED.jsonl"

# ---------------- LOAD MANUAL TOPICS ----------------
cluster_to_topic = {}

used_ids = set()
current_id = None

with open(MANUAL_TOPICS_TXT, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue

        m = re.match(r"^(\d+)\s+(.*)$", line)
        if not m:
            print(f"[WARN] Skipping line {line_num}: {line}")
            continue

        raw_id = int(m.group(1))
        raw_topic = m.group(2).strip()

        # 🔪 REMOVE EVERYTHING IN ()
        clean_topic = re.split(r"\s*\(", raw_topic, maxsplit=1)[0].strip()

        # Fix duplicated 27 → shift forward
        if current_id is None:
            fixed_id = raw_id
        else:
            if raw_id in used_ids:
                fixed_id = current_id + 1
            else:
                fixed_id = raw_id

        used_ids.add(fixed_id)
        current_id = fixed_id

        cluster_to_topic[fixed_id] = clean_topic

print(f"Loaded {len(cluster_to_topic)} manual topics")

# ---------------- REWRITE JSONL ----------------
count_manual = 0
count_razno = 0
count_untouched = 0

with open(INPUT_JSONL, "r", encoding="utf-8") as fin, \
     open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:

    for line_num, line in enumerate(fin, 1):
        obj = json.loads(line)
        cluster_id = obj.get("cluster")

        if cluster_id == -999:
            obj["cluster_topic"] = "razno"
            count_razno += 1

        elif cluster_id in cluster_to_topic:
            obj["cluster_topic"] = cluster_to_topic[cluster_id]
            count_manual += 1

        else:
            count_untouched += 1

        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

print("✅ DONE")
print(f"✔ Manual topics applied : {count_manual}")
print(f"✔ Razno applied         : {count_razno}")
print(f"⚠ Untouched rows       : {count_untouched}")
print(f"📄 Output file          : {OUTPUT_JSONL}")
