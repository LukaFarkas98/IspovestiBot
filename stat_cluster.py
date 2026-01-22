import json
from collections import defaultdict

INPUT_FILE = "confessions_clusters_INCLUDE_MORE_DATA.jsonl"
OUTPUT_FILE = "clusters_preview.txt"
MAX_PER_CLUSTER = 10

clusters = defaultdict(list)

# Read JSONL
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Skipping line {line_num}: JSON error -> {e}")
            continue

        cluster_id = obj.get("cluster")
        text = obj.get("text", "").strip()
        topic = obj.get("cluster_topic", "").strip()

        if cluster_id is None:
            continue

        if len(clusters[cluster_id]) < MAX_PER_CLUSTER:
            clusters[cluster_id].append({
                "text": text,
                "topic": topic
            })

# Write output
with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for cluster_id in sorted(clusters.keys()):
        out.write("=" * 40 + "\n")
        out.write(f"CLUSTER ID: {cluster_id}\n")
        out.write("=" * 40 + "\n")

        for i, entry in enumerate(clusters[cluster_id], 1):
            out.write(f"\n[{i}]\n")
            if entry["topic"]:
                out.write(f"Topic: {entry['topic']}\n")
            out.write(entry["text"] + "\n")

        out.write("\n\n")

print(f"Done. Output written to: {OUTPUT_FILE}")
