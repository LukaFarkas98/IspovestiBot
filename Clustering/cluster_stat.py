import json
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np       

INPUT_FILE = "confessions_with_clusters_hybrid.jsonl"


import json
from collections import defaultdict

cluster_texts = defaultdict(list)


#PRESLOZI OVO U JEDNU FUNKCIJU DA DOBIJEMO KLJUCNE RECI ZA KLASTERE
def printNumbersByCluster():
    counter = Counter()
    with open("confessions_with_clusters_hybrid.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            counter[json.loads(line)["cluster_id"]] += 1

    for cid, count in counter.most_common():
        print(f"Cluster {cid}: {count} texts")


with open("confessions_with_clusters_hybrid.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        cid = item["cluster_id"]
        if cid != -1:  # skip noise
            cluster_texts[cid].append(item["text"])

     
def top_keywords_per_cluster(texts, top_k=10):
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=5
    )
    X = vectorizer.fit_transform(texts)
    scores = np.asarray(X.mean(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    top_idx = scores.argsort()[::-1][:top_k]
    return [terms[i] for i in top_idx]

cluster_sizes = Counter({cid: len(txts) for cid, txts in cluster_texts.items()})

for cid, size in cluster_sizes.most_common(10):
    keywords = top_keywords_per_cluster(cluster_texts[cid], top_k=8)
    print(f"\nCluster {cid} ({size} texts)")
    print(", ".join(keywords))

