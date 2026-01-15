import json
import numpy as np
from collections import defaultdict

from sentence_transformers import SentenceTransformer
import hdbscan
import umap
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


# =====================
# CONFIG
# =====================

INPUT_FILE = "confessions_Archive_140k_clean.jsonl"
OUTPUT_FILE = "confessions_with_clusters_hybrid_140K.jsonl"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 64

UMAP_COMPONENTS = 20
UMAP_NEIGHBORS = 15

MIN_CLUSTER_SIZE = 10
MIN_SAMPLES = 3


# =====================
# LOAD DATA
# =====================

texts = []
meta = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        texts.append(item["text"])
        meta.append(item)

print(f"Loaded {len(texts)} confessions")


# =====================
# EMBEDDINGS
# =====================

model = SentenceTransformer(MODEL_NAME)

print("Encoding texts...")
embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True,
)


# =====================
# UMAP
# =====================

print("Reducing dimensions...")
reduced = umap.UMAP(
    n_neighbors=UMAP_NEIGHBORS,
    n_components=UMAP_COMPONENTS,
    metric="cosine",
).fit_transform(embeddings)


# =====================
# HDBSCAN
# =====================

print("Running HDBSCAN...")
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=MIN_CLUSTER_SIZE,
    min_samples=MIN_SAMPLES,
    metric="euclidean",
    core_dist_n_jobs=-1,
)

labels = clusterer.fit_predict(reduced)


# =====================
# COMPUTE CLUSTER CENTROIDS (EMBEDDING SPACE!)
# =====================

print("Computing cluster centroids...")

cluster_to_vectors = defaultdict(list)

for label, emb in zip(labels, embeddings):
    if label != -1:
        cluster_to_vectors[label].append(emb)

cluster_centroids = {
    label: np.mean(vectors, axis=0)
    for label, vectors in cluster_to_vectors.items()
}

cluster_ids = list(cluster_centroids.keys())
centroid_matrix = np.vstack([cluster_centroids[c] for c in cluster_ids])


# =====================
# ASSIGN NOISE POINTS
# =====================

print("Assigning noise points to nearest cluster...")

final_labels = labels.copy()

for i, label in tqdm(enumerate(labels), total=len(labels)):
    if label == -1:
        sims = cosine_similarity(
            embeddings[i].reshape(1, -1),
            centroid_matrix
        )
        best_cluster = cluster_ids[np.argmax(sims)]
        final_labels[i] = best_cluster


# =====================
# SAVE RESULTS
# =====================

print("Saving hybrid clustered dataset...")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for item, label in zip(meta, final_labels):
        item["cluster_id"] = int(label)
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("Done!")
