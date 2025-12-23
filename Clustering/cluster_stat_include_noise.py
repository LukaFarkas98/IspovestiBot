import json
import random
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

INPUT_FILE = "confessions_with_clusters.jsonl"
OUTPUT_FILE = "clusters_with_NOISE.json"

SAMPLE_PER_CLUSTER = 5
TOP_K_KEYWORDS = 10
MIN_TEXTS_FOR_TFIDF = 20

SERBIAN_STOPWORDS = {
    "da","je","sam","se","mi","ne","na","za","su","ali","me","koji","od","nije",
    "to","sve","samo","ili","što","sa","ja","kad","kako","jer","kada","znam",
    "ima","iz","mu","smo","rođendan","ispovest","ispovijest"
}

# -------------------------
# Load texts per cluster
# -------------------------
cluster_texts = defaultdict(list)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        cid = str(item["cluster_id"])   # keep as string for JSON keys
        cluster_texts[cid].append(item["text"])

print(f"Loaded {len(cluster_texts)} clusters (including noise)")

# -------------------------
# Keyword extraction
# -------------------------
def extract_keywords(texts, top_k=10):
    if len(texts) < MIN_TEXTS_FOR_TFIDF:
        return []

    vectorizer = TfidfVectorizer(
        stop_words=list(SERBIAN_STOPWORDS),
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.8,
        max_features=8000
    )

    try:
        X = vectorizer.fit_transform(texts)
    except ValueError:
        return []

    if X.shape[1] == 0:
        return []

    scores = np.asarray(X.mean(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    top_idx = scores.argsort()[::-1][:top_k]

    return [terms[i] for i in top_idx]

# -------------------------
# Build cluster summary
# -------------------------
cluster_summary = {}

for cid, texts in cluster_texts.items():
    samples = random.sample(texts, min(SAMPLE_PER_CLUSTER, len(texts)))

    cluster_summary[cid] = {
        "size": len(texts),
        "keywords": extract_keywords(texts, TOP_K_KEYWORDS),
        "sample_confessions": samples
    }

# -------------------------
# Save output
# -------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(cluster_summary, f, ensure_ascii=False, indent=2)

print(f"Saved cluster inspection file → {OUTPUT_FILE}")
