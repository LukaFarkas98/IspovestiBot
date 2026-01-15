import json
import numpy as np
from collections import defaultdict

from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# CONFIG
# =========================
INPUT_FILE = "confessions_with_clusters_hybrid_140K.jsonl"
CLUSTER_SUMMARY_FILE = "clusters_summary.json"

OUTPUT_META_FILE = "meta_clusters.json"
OUTPUT_MAPPING_FILE = "cluster_to_meta_topic.json"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Tune this
TARGET_META_CLUSTERS = 150  # try 80–150

# =========================
# STEP 1: LOAD DATA
# =========================
print("📥 Loading clustered dataset...")

cluster_texts = defaultdict(list)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        cluster_texts[item["cluster_id"]].append(item["text"])

cluster_ids = sorted(cluster_texts.keys())
print(f"🧠 Found {len(cluster_ids)} base clusters")

# =========================
# STEP 2: COMPUTE CENTROIDS
# =========================
print("🔢 Computing cluster centroids...")

model = SentenceTransformer(MODEL_NAME)

cluster_centroids = []
valid_cluster_ids = []

for cid in cluster_ids:
    texts = cluster_texts[cid]
    if len(texts) < 5:
        continue  # too small, skip for stability

    emb = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    centroid = np.mean(emb, axis=0)
    cluster_centroids.append(centroid)
    valid_cluster_ids.append(cid)

X = np.vstack(cluster_centroids)

print(f"✅ Centroids computed for {len(valid_cluster_ids)} clusters")

# =========================
# STEP 3: META-CLUSTERING
# =========================
print("🧩 Running meta-clustering...")

meta_clusterer = AgglomerativeClustering(
    n_clusters=TARGET_META_CLUSTERS,
    metric="cosine",
    linkage="average"
)

meta_labels = meta_clusterer.fit_predict(X)

# =========================
# STEP 4: BUILD META TOPICS
# =========================
print("🏷️ Building meta-topics...")

meta_to_clusters = defaultdict(list)

for cid, meta_id in zip(valid_cluster_ids, meta_labels):
    meta_to_clusters[int(meta_id)].append(int(cid))

# Load keywords for naming
with open(CLUSTER_SUMMARY_FILE, "r", encoding="utf-8") as f:
    cluster_summary = json.load(f)

cluster_id_to_keyword = {
    c["cluster_id"]: c["keyword"]
    for c in cluster_summary
}

meta_topics = {}
cluster_to_meta = {}

##



##
for meta_id, cids in meta_to_clusters.items():
    keywords = [
        cluster_id_to_keyword.get(cid, "")
        for cid in cids
    ]

    # crude but effective: most common token across keywords
    token_freq = defaultdict(int)
    for kw in keywords:
        for token in kw.split("_"):
            if token:
                token_freq[token] += 1

    top_tokens = sorted(
        token_freq.items(),
        key=lambda x: x[1],
        reverse=True
    )[:3]

    meta_name = "_".join(t[0] for t in top_tokens)
    meta_label = f"TOPIC_{meta_name.upper()}" if meta_name else f"TOPIC_{meta_id}"

    meta_topics[meta_id] = {
        "meta_topic": meta_label,
        "num_clusters": len(cids),
        "clusters": cids
    }

    for cid in cids:
        cluster_to_meta[cid] = meta_label

# =========================
# STEP 5: SAVE OUTPUTS
# =========================
with open(OUTPUT_META_FILE, "w", encoding="utf-8") as f:
    json.dump(meta_topics, f, ensure_ascii=False, indent=2)

with open(OUTPUT_MAPPING_FILE, "w", encoding="utf-8") as f:
    json.dump(cluster_to_meta, f, ensure_ascii=False, indent=2)

print("🎉 Meta-clustering done!")
print(f"📄 Meta topics → {OUTPUT_META_FILE}")
print(f"🔗 Cluster → meta mapping → {OUTPUT_MAPPING_FILE}")
