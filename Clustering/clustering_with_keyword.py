import json
import re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

# =========================
# CONFIG
# =========================
INPUT_FILE = "confessions_with_clusters_hybrid_140K.jsonl"
OUTPUT_FILE = "confessions_for_training_with_KEYWORDS_140K.jsonl"
CLUSTER_SUMMARY_FILE = "clusters_summary.json"

TOP_K_KEYWORDS = 3
SAMPLE_TEXTS_PER_CLUSTER = 3

SERBIAN_STOPWORDS = [
    "i", "u", "na", "se", "je", "sam", "su", "sa", "za", "što", "ali",
    "od", "do", "koji", "kada", "će", "ne", "sve", "ili", "ako", "ja",
    "ti", "on", "ona", "oni", "one", "mi", "vi", "bio", "da", "je",
    "nije", "to", "samo", "kad", "kako", "jer", "znam", "ima", "iz",
    "smo", "bi", "bih", "bila", "biti", "bude", "bilo", "ni",
    "danas", "dan", "gde", "ovde", "imam", "godina", "godine", "ispovest", "me", "šta",
    "kao", "pa", "ga", "joj", "kod", "joj"
]

NOISE_CLUSTER_ID = -1

# =========================
# STEP 1: LOAD + GROUP
# =========================
cluster_texts = defaultdict(list)
raw_items = []

print("📥 Loading dataset...")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue

        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            print(f"[WARN] Bad JSON on line {line_num}")
            continue

        if "text" not in item or "cluster_id" not in item:
            continue

        cluster_id = int(item["cluster_id"])
        text = item["text"].strip()

        cluster_texts[cluster_id].append(text)
        raw_items.append((cluster_id, text))

print(f"✅ Loaded {len(raw_items)} texts")
print(f"🧠 Found {len(cluster_texts)} clusters (including noise)")

# =========================
# STEP 2: COMPUTE KEYWORDS
# =========================
cluster_keywords = {}

print("🔑 Computing keywords per cluster...")

for cluster_id, texts in cluster_texts.items():
    if cluster_id == NOISE_CLUSTER_ID:
        cluster_keywords[cluster_id] = "NO_TOPIC"
        continue

    vectorizer = TfidfVectorizer(
        max_features=100,
        stop_words=SERBIAN_STOPWORDS,
        ngram_range=(1, 2),
        min_df=2
    )

    X = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()

    scores = X.mean(axis=0).A1
    top_idx = scores.argsort()[::-1][:TOP_K_KEYWORDS]
    top_terms = [terms[i] for i in top_idx]

    keyword = "_".join(top_terms) if top_terms else f"TOPIC_{cluster_id}"
    cluster_keywords[cluster_id] = keyword

print("🧪 Sample keywords:")
for cid in list(cluster_keywords.keys())[:10]:
    print(cid, "→", cluster_keywords[cid])

# =========================
# STEP 3: CLUSTER SUMMARY
# =========================
print("📊 Building cluster summary...")

cluster_summary = []

for cluster_id, texts in cluster_texts.items():
    cluster_summary.append({
        "cluster_id": cluster_id,
        "keyword": cluster_keywords.get(cluster_id, "UNKNOWN"),
        "num_texts": len(texts),
        "sample_texts": texts[:SAMPLE_TEXTS_PER_CLUSTER]
    })

cluster_summary.sort(key=lambda x: x["num_texts"], reverse=True)

with open(CLUSTER_SUMMARY_FILE, "w", encoding="utf-8") as f:
    json.dump(cluster_summary, f, ensure_ascii=False, indent=2)

print(f"✅ Cluster summary saved → {CLUSTER_SUMMARY_FILE}")

# =========================
# STEP 4: REWRITE DATASET
# =========================
print("✍️ Rewriting dataset with keyword tags...")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for cluster_id, text in raw_items:
        keyword = cluster_keywords.get(cluster_id, "UNKNOWN")
        new_text = f"<{keyword}> Ispovest:\n{text} <{keyword}>"
        f.write(json.dumps({"text": new_text}, ensure_ascii=False) + "\n")

print(f"🎉 Done! Output saved → {OUTPUT_FILE}")
