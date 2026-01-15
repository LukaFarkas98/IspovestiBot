import json
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
import hdbscan
from keybert import KeyBERT
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import umap
import modal
import time

# -----------------------
# Modal App
# -----------------------
volumeName = "my-volume"
volume = modal.Volume.from_name(volumeName)
app = modal.App("ispovesti-clustering-umap")

# -----------------------
# Paths / constants
# -----------------------
EMBEDDINGS_PATH = "/mnt/models/embeddings.npy"
REDUCED_PATH = "/mnt/models/reduced_umap.npy"
INPUT_JSONL = "/mnt/models/confessions_COMBINED_140K.jsonl"
OUTPUT_JSONL = "/mnt/models/confessions_clusters_INCLUDE_MORE_DATA.jsonl"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

UMAP_DIMS = 10  # or 15

MIN_CLUSTER_SIZE = 20  # or 20
MIN_SAMPLES = 5  
KEYWORDS_PER_CLUSTER = 5
DEVICE = "cuda"
BATCH_SIZE = 20000  # for batched HDBSCAN

# -----------------------
# Safe JSONL loader
# -----------------------
def safe_jsonl_loader(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

# -----------------------
# Batched HDBSCAN
# -----------------------
def hdbscan_cluster_soft(embeddings, prob_threshold=0.3):
    print(f"⏳ HDBSCAN: clustering all {embeddings.shape[0]} points at {time.strftime('%H:%M:%S')}")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="leaf",
        core_dist_n_jobs=-1
    )
    labels = clusterer.fit_predict(embeddings)
    probs = clusterer.probabilities_  # probability of cluster membership

    # Assign low-confidence points (-1) to their closest cluster if prob > threshold
    labels_soft = labels.copy()
    for i, (label, prob) in enumerate(zip(labels, probs)):
        if label == -1 and prob >= prob_threshold:
            # Assign to the cluster with highest soft membership
            labels_soft[i] = clusterer.labels_[clusterer.single_linkage_tree_.get_leaf(i)[0]]

    n_clusters = len(set(labels_soft)) - (1 if -1 in labels_soft else 0)
    print(f"Clusters found (soft assignment): {n_clusters}")
    return labels_soft

# -----------------------
# Pipeline
# -----------------------
@app.function(
    gpu="A100-40GB",
    timeout=60 * 60 * 4,
    memory=32000,
    volumes={"/mnt/models": volume},
    image=modal.Image.debian_slim().pip_install([
        "torch",
        "transformers",
        "sentence-transformers",
        "keybert",
        "hdbscan",
        "umap-learn",
        "tqdm",
        "numpy",
        "accelerate"
    ]),
)
def run_pipeline():
    # -------- LOAD DATA --------
    data = list(safe_jsonl_loader(INPUT_JSONL))
    texts = [d["text"] for d in data]
    print(f"Loaded {len(texts)} confessions")

    # -------- EMBEDDINGS --------
    embedder = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
    embeddings = embedder.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    np.save(EMBEDDINGS_PATH, embeddings)
    print("Computed embeddings:", embeddings.shape)
    
    # -------- UMAP --------
    reducer = umap.UMAP(n_components=UMAP_DIMS, n_neighbors=30, min_dist=0.0, random_state=42)
    reduced = reducer.fit_transform(embeddings)
    np.save(REDUCED_PATH, reduced)
    print("Computed UMAP:", reduced.shape)
    
    # -------- HDBSCAN CLUSTERING --------
    labels_soft = hdbscan_cluster_soft(reduced, prob_threshold=0.3)
    

    # -------- GROUP BY CLUSTER --------
    clusters = {}
    indices = {}
    for i, label in enumerate(labels_soft):
        if label == -1:
            continue
        clusters.setdefault(label, []).append(texts[i])
        indices.setdefault(label, []).append(i)

    # -------- KEYBERT --------
    kw_model = KeyBERT(embedder)
    candidates = {}
    for label, docs in tqdm(clusters.items(), desc="KeyBERT"):
        sample_docs = docs[:30] if len(docs) > 30 else docs
        joined = " ".join(sample_docs)
        kws = kw_model.extract_keywords(
            joined,
            top_n=15,
            use_mmr=True,
            diversity=0.5
        )
        candidates[label] = [k for k, _ in kws]

    # -------- LLM TOPIC LABELS --------
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        device_map="auto",
        dtype="auto"
    )
    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=24,
        temperature=0.3
    )

    topics = {}
    for label, kws in tqdm(candidates.items(), desc="LLM topics"):
        prompt = f"""
Na osnovu ključnih reči daj KRATAK naziv teme (1–3 reči, srpski).

Ključne reči:
{", ".join(kws)}

Tema:
"""
        out = gen(prompt)[0]["generated_text"]
        topic = out.split("Tema:")[-1].strip()
        topics[label] = topic or kws[0]

    # -------- SAVE JSONL --------
        with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
            for i, label in enumerate(labels_soft):
                # If still -1 after soft assignment, assign them to a dummy cluster
                cluster_id = int(label) if label != -1 else -999
                f.write(json.dumps({
                    "text": texts[i],
                    "cluster": cluster_id,
                    "cluster_topic": topics.get(cluster_id, "Misc")  # fallback topic
                }, ensure_ascii=False) + "\n")

    print("✅ Saved to", OUTPUT_JSONL)
    print("Example topics:", {k: topics[k] for k in list(topics)[:5]})
    return OUTPUT_JSONL

# -----------------------
# Local
# -----------------------
@app.local_entrypoint()
def main():
    print("Deploying clustering + UMAP + HDBSCAN job on Modal GPU...")
    app.deploy()
    run_id = run_pipeline.spawn()
    print("Run ID:", run_id)
