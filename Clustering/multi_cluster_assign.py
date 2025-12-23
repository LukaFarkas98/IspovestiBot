import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

INPUT_FILE = "confessions_with_clusters_hybrid.jsonl"
OUTPUT_FILE = "confessions_with_multiclusters.jsonl"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
N_CLUSTERS = 3000
TOP_K = 3   # how many clusters per confession

# -------------------------
# Load texts
# -------------------------
texts = []
raw_items = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        raw_items.append(item)
        texts.append(item["text"])

print(f"Loaded {len(texts)} confessions")

# -------------------------
# Encode
# -------------------------
model = SentenceTransformer(EMBEDDING_MODEL)
embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)

# -------------------------
# Fit clustering
# -------------------------
print("Clustering...")
kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=42,
    n_init=10
)
kmeans.fit(embeddings)

centroids = kmeans.cluster_centers_

# -------------------------
# Assign MULTIPLE clusters
# -------------------------
print("Assigning multiple clusters...")

similarities = cosine_similarity(embeddings, centroids)

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for item, sims in tqdm(zip(raw_items, similarities), total=len(raw_items)):
        top_clusters = np.argsort(sims)[::-1][:TOP_K]

        item["cluster_ids"] = top_clusters.tolist()
        item["cluster_scores"] = sims[top_clusters].tolist()

        out.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Saved multi-cluster labels to {OUTPUT_FILE}")
