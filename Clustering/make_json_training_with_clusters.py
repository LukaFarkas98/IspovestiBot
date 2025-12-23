import json

INPUT_FILE = "confessions_with_clusters.jsonl"
OUTPUT_FILE = "confessions_for_training_with_TOPICS.jsonl"

with open(INPUT_FILE, "r", encoding="utf-8") as f_in, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:

    for line in f_in:
        item = json.loads(line)
        cluster_id = item.get("cluster_id", -1)

        text = item["text"].strip()

        if cluster_id == -1:
            # Noise → generic topic
            prefixed_text = f"<NO_TOPIC> {text} <NO_TOPIC>"
        else:
            prefixed_text = f"<CLUSTER_{cluster_id}> {text} <CLUSTER_{cluster_id}>"

        f_out.write(json.dumps({"text": prefixed_text}, ensure_ascii=False) + "\n")

print(f"Training JSONL saved to {OUTPUT_FILE}")
