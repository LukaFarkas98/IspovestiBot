import json
import random

# -----------------------
# SETTINGS
# -----------------------
INPUT_JSONL = "confessions_FINAL_WITH_ENGAGEMENT.jsonl"
OUTPUT_JSONL = "confessions_for_training.jsonl"
SAMPLE_RAZNO = 5000  # number of razno confessions to include
EXTRA_TOPICS = 1     # simulate extra topics per confession for multi-topic learning

# -----------------------
# Load all confessions
# -----------------------
clustered = []
razno = []

all_topics = set()

with open(INPUT_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        topic = obj["cluster_topic"].strip().lower()
        if topic:
            all_topics.add(topic)

        if obj["cluster"] == -999 or obj["cluster_topic"].lower() == "razno":
            razno.append(obj)
        else:
            clustered.append(obj)

all_topics = list(all_topics)
print(f"Found {len(clustered)} clustered confessions and {len(razno)} razno")
print(f"Total unique topics: {len(all_topics)}")

# -----------------------
# Sample razno
# -----------------------
sampled_razno = random.sample(razno, min(SAMPLE_RAZNO, len(razno)))

# -----------------------
# Combine and shuffle
# -----------------------
training_data = clustered + sampled_razno
random.shuffle(training_data)

# -----------------------
# Prepend engagement score + topic tokens
# -----------------------
processed = []
for obj in training_data:
    main_topic = obj["cluster_topic"].strip().lower()

    # choose extra random topics for multi-topic learning
    extra = []
    if EXTRA_TOPICS > 0:
        candidates = [t for t in all_topics if t != main_topic]
        extra = random.sample(candidates, min(EXTRA_TOPICS, len(candidates)))

    # create topic tokens
    topic_tokens = " ".join([f"[{main_topic}]"] + [f"[{t}]" for t in extra])

    # engagement score token
    score_token = f"[engagement_score:{obj.get('engagement_score', 0.0):.4f}]"

    # final text
    new_text = f"{score_token} {topic_tokens} Ispovest:\n{obj['text'].replace('Ispovest:\n','')}"
    obj["text"] = new_text

    processed.append(obj)

# -----------------------
# Write final JSONL
# -----------------------
with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for item in processed:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ Finished preprocessing. Total confessions: {len(processed)}")
print(f"Output file: {OUTPUT_JSONL}")
