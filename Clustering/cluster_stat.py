import json
import random
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

INPUT_FILE = "confessions_with_clusters_hybrid.jsonl"
OUTPUT_JSON = "cluster_inspection.json"

SERBIAN_STOPWORDS = {
    "da","je","sam","se","mi","ne","na","rođendan","ispovest","ispovijest",
    "za","su","ali","me","koji","od","nije","to","sve","samo","ili","što","sa",
    "ja","kad","kako","jer","kada","znam","ima","iz","mu","smo"
}

# -------------------------
# Load texts per cluster
# -------------------------
cluster_texts = defaultdict(list)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        cid = item["cluster_id"]
        if cid != -1:
            cluster_texts[cid].append(item["text"])


# -------------------------
# TF-IDF keywords
# -------------------------
def top_keywords_per_cluster(texts, top_k=10):
    if len(texts) < 20:
        return []

    vectorizer = TfidfVectorizer(
        max_features=8000,
        stop_words=list(SERBIAN_STOPWORDS),
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.8
    )

    try:
        X = vectorizer.fit_transform(texts)
    except ValueError:
        return []

    if X.shape[1] == 0:
        return []

    scores = np.asarray(X.mean(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()

    if len(terms) == 0:
        return []

    top_idx = scores.argsort()[::-1][:top_k]
    return [terms[i] for i in top_idx]


# -------------------------
# Build cluster summary
# -------------------------
cluster_summary = {}

for cid, texts in cluster_texts.items():
    sample_size = min(5, len(texts))
    random_samples = random.sample(texts, sample_size)

    cluster_summary[cid] = {
        "size": len(texts),
        "keywords": top_keywords_per_cluster(texts, top_k=10),
        "sample_confessions": random_samples
    }

# -------------------------
# Save to JSON
# -------------------------
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(cluster_summary, f, ensure_ascii=False, indent=2)

print(f"Saved cluster inspection data for {len(cluster_summary)} clusters to {OUTPUT_JSON}")
