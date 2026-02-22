# ARCHITECTURE.md ¡X End-to-End NLP Pipeline (20 Newsgroups)

This repo implements a **3-part, end-to-end NLP pipeline** on the **scikit-learn 20 Newsgroups** dataset for:
1) **Multi-class supervised classification** with **classic sparse features (BoW / TF-IDF)**  
2) **Multi-class supervised classification** with **SentenceTransformer embeddings**  
3) **Semantic clustering (<10 clusters)** + **2-level topic tree** with **LLM-generated labels** (via ¡§packet¡¨ files)

The implementation is designed to be **reproducible, leakage-safe (Pipelines), and artifact-driven** (metrics CSV, confusion analysis, clustering summaries, topic tree text, and LLM prompt packets).

---

## 1) System Overview

### Core data flow (high level)

**Dataset ¡÷ minimal cleaning ¡÷ representation ¡÷ model ¡÷ evaluation ¡÷ artifacts**

- **Dataset**: `fetch_20newsgroups` official **train/test split** (>=10k docs, >5 classes; 20 classes). :contentReference[oaicite:0]{index=0}  
- **Cleaning**: whitespace normalization + strip; intentionally minimal to avoid overfitting to heavy preprocessing. :contentReference[oaicite:1]{index=1}  
- **Representation**:
  - Part 1: `CountVectorizer` or `TfidfVectorizer` (sparse) :contentReference[oaicite:2]{index=2}
  - Part 2/3: `SentenceTransformer` embeddings (dense) with optional L2 normalization + caching :contentReference[oaicite:3]{index=3}  
- **Models** (same set for Part 1 & Part 2):
  - Multinomial Naive Bayes (MNB)
  - Logistic Regression
  - Linear SVM (LinearSVC)
  - Random Forest   
- **Evaluation**: Accuracy + Macro-F1 + confusion matrix + top confusion pairs :contentReference[oaicite:5]{index=5}  
- **Clustering** (Part 3):
  - KMeans top-level **K < 10** with elbow plot and max-curvature heuristic
  - Re-cluster the **two largest clusters** into **exactly 3 subclusters each**
  - Create a **2-level topic tree** in `outputs/part3/topic_tree.txt`
  - Produce **LLM ¡§label packets¡¨** for manual labeling (or TF-IDF fallback labels)   

---

## 2) Repository Components

### Scripts (entrypoints)

- `scripts/run_part1.py`  
  Runs classic BoW/TF-IDF pipelines and exports metrics + confusion artifacts. :contentReference[oaicite:7]{index=7}

- `scripts/run_part2.py`  
  Embeds train/test, runs classifiers on embeddings, exports metrics + confusion artifacts, and prints a Part1-vs-Part2 comparison when Part1 outputs exist. :contentReference[oaicite:8]{index=8}

- `scripts/run_part3.py`  
  Embeds **all docs** (train+test), does elbow selection (K in [2..9]), top-level clustering, subclustering of the top 2 clusters, outputs summaries + topic tree, and writes LLM labeling packets. :contentReference[oaicite:9]{index=9}

### Library modules (`src/`)

- `src/data_loader.py`  
  Loads 20 Newsgroups train/test, applies minimal cleaning, returns texts/labels/target_names. :contentReference[oaicite:10]{index=10}

- `src/part1_classic.py`  
  Defines Part1 models and builds the leakage-safe sklearn `Pipeline(vectorizer ¡÷ model)`. :contentReference[oaicite:11]{index=11}

- `src/part2_embeddings.py`  
  Defines Part2 models and embedding-time pipelines (LR/SVM include `StandardScaler`; RF does not; MNB included for completeness). :contentReference[oaicite:12]{index=12}

- `src/embedding_cache.py`  
  Caches SentenceTransformer embeddings on disk using a stable hash of the embedding configuration (model name, split, remove flags, batch_size, seed, normalize). :contentReference[oaicite:13]{index=13}

- `src/metrics.py`  
  Measures Accuracy + Macro-F1 + confusion matrix; saves confusion PNG; extracts top confusion pairs. :contentReference[oaicite:14]{index=14}

- `src/cluster_utils.py`  
  Elbow inertia, K selection (max-curvature heuristic), KMeans fit, centroid-based representatives, and **contrastive TF-IDF fallback cluster labeling**. :contentReference[oaicite:15]{index=15}

- `src/llm_packets.py`  
  Writes prompt ¡§packets¡¨ (plain text files) to feed into an LLM. The LLM must return **STRICT JSON** with label/rationale/keywords. :contentReference[oaicite:16]{index=16}

- `src/cluster_viz.py` (optional)  
  UMAP visualization helper (requires `umap-learn`). :contentReference[oaicite:17]{index=17}

### Dependencies

See `requirements.txt`. Notably:
- `scikit-learn`, `numpy`, `pandas`, `matplotlib`
- `sentence-transformers`, `torch`
- `umap-learn` (optional, only if you enable UMAP plots) :contentReference[oaicite:18]{index=18}

---

## 3) Part 1 ¡X Classic Features (BoW / TF-IDF) Classification

### Architecture

**Input**: raw post text  
**Processing**:
1) Load train/test split and minimally clean text :contentReference[oaicite:19]{index=19}  
2) Build sklearn `Pipeline(vectorizer ¡÷ model)` to prevent leakage   
3) Train and evaluate each of the 4 classifiers   
4) Select best model by **Macro-F1** and export confusion artifacts   

**Vectorization knobs** (CLI-configurable):
- `--vectorizer {bow, tfidf}`
- `--max_features` (default 60000)
- `--ngram_max {1,2}`
- `--min_df`
- `--stop_words {none, english}`
- `--remove headers,footers,quotes` (removes metadata noise)   

### Outputs (Part 1)

- `outputs/part1_metrics.csv` (required exact path) :contentReference[oaicite:24]{index=24}  
- `outputs/part1/top_confusion_pairs.json` (best model)   
- optional: `outputs/part1/confusion_matrix_best_<MODEL>.png`   
- `outputs/part1/run_metadata.json` (reproducibility snapshot) :contentReference[oaicite:27]{index=27}  

---

## 4) Part 2 ¡X SentenceTransformer Embeddings + Classical Classifiers

### Architecture

**Input**: raw post text  
**Processing**:
1) Load train/test split and minimally clean text   
2) Encode documents with SentenceTransformer; cache embeddings to disk   
3) Train the same classifier set on embeddings:
   - LR/SVM use `StandardScaler(with_mean=True)` because dense feature scales matter :contentReference[oaicite:30]{index=30}  
   - RandomForest runs directly
   - MNB is attempted but typically skipped/underperforms because embeddings can be negative and aren¡¦t count-like   
4) Evaluate on held-out test (Accuracy + Macro-F1 + confusion)   
5) Print an automatic Part1-vs-Part2 comparison if Part1 metrics exist in `outputs/` :contentReference[oaicite:33]{index=33}  

**Embedding knobs**:
- `--st_model all-MiniLM-L6-v2` (default)
- `--batch_size` (default 64)
- `--normalize` (optional L2 normalization)
- `--cache_dir outputs/cache`   

### Outputs (Part 2)

- `outputs/part2_metrics.csv` :contentReference[oaicite:35]{index=35}  
- `outputs/part2/top_confusion_pairs.json`   
- optional: `outputs/part2/confusion_matrix_best_<MODEL>.png`   
- `outputs/part2/run_metadata.json` :contentReference[oaicite:38]{index=38}  
- Embedding cache files under `outputs/cache/*.npz` + `*.json` meta   

---

## 5) Part 3 ¡X Topic Clustering + 2-Level Topic Tree

### Architecture

**Goal**: meaningful **<10** top-level clusters + a **2-level topic tree** with subclusters for the two largest clusters.

**Step A ¡X Top-level clustering (<10 clusters)**   
1) Load all docs (train+test) ¡÷ `texts_all`   
2) Embed all docs (cached)   
3) Run elbow method for K in `[2..9]`, save `part3_elbow.png`   
4) Choose K:
   - default: **max-curvature heuristic** on inertia curve
   - optional: `--k_override` if you want full manual control   
5) Fit KMeans(K) on embeddings   
6) For each cluster:
   - Extract centroid-nearest representative docs (snippets)   
   - Generate a **fallback label** using **contrastive TF-IDF** keywords (distinctive terms)   
   - If `--label_mode packet`, write an LLM labeling packet file   

**Step B ¡X Second-level clustering on 2 biggest clusters** :contentReference[oaicite:49]{index=49}  
1) Find the top 2 clusters by size  
2) For each parent cluster:
   - Re-cluster into **exactly 3** subclusters (`k=3`)
   - Create reps + fallback labels
   - Write subcluster packet files (if packet mode)   

**Step C ¡X Show partial tree** :contentReference[oaicite:51]{index=51}  
- Build and save `outputs/part3/topic_tree.txt` using fallback labels
- Your final submission replaces fallback labels with LLM labels after you run the packets.

### ¡§Labeling mode¡¨ design (packet vs fallback)

- `--label_mode packet` (default):  
  Produces both:
  - fallback TF-IDF label (for immediate tree preview)
  - packet files in `outputs/part3/llm_packets/` to paste into ChatGPT and collect STRICT JSON responses   

- `--label_mode fallback`:  
  Skips packet generation; uses only TF-IDF fallback labels   

### Outputs (Part 3)

All outputs live under `outputs/part3/` :contentReference[oaicite:54]{index=54}  

- `part3_elbow.png` (K selection evidence)   
- `part3_top_clusters.json` (cluster summaries: size, fallback label, keywords, reps, packet path)   
- `part3_subclusters.json` (same for subclusters)   
- `topic_tree.txt` (2-level tree view) :contentReference[oaicite:58]{index=58}  
- `llm_packets/cluster_<id>_label_packet.txt` and `cluster_<id>_sub<sid>_label_packet.txt`   
- optional (if `--umap`): `cluster_scatter_umap_top.png` and `cluster_scatter_umap_sub_parent<cid>.png`   

---

## 6) Running the Pipeline (Commands)

> All scripts are designed to be run from repo root.

### Part 1
```bash
python scripts/run_part1.py --vectorizer tfidf --ngram_max 2 --save_confusion_png