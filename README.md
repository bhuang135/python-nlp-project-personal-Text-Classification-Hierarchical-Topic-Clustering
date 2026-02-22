News20 NLP Pipeline



End-to-end NLP pipeline on the 20 Newsgroups dataset including:



Part 1 — TF-IDF / BoW classification



Part 2 — SentenceTransformer embedding classification



Part 3 — KMeans clustering + hierarchical topic tree



============================================================

QUICK START (SETUP)



Create virtual environment



python -m venv .venv



Activate:



Windows:

.venv\\Scripts\\activate



Mac/Linux:

source .venv/bin/activate



Install dependencies



pip install -r requirements.txt



(Optional for UMAP visualization)

pip install umap-learn



============================================================

HOW TO RUN

Part 1 — TF-IDF / BoW Classification



Run with TF-IDF:



python -m scripts.run\_part1 --vectorizer tfidf --save\_confusion\_png



Run with Bag-of-Words:



python -m scripts.run\_part1 --vectorizer bow --save\_confusion\_png



Outputs saved in:

outputs/part1/



Part 2 — SentenceTransformer Classification



Basic run:



python -m scripts.run\_part2 --st\_model all-MiniLM-L6-v2



Optional flags:



--normalize

--batch\_size 64



Outputs saved in:

outputs/part2/



Note:

Embeddings are cached in:

outputs/cache/

If cache exists, embeddings are not recomputed.



Part 3 — Clustering + Topic Tree



Basic run:



python -m scripts.run\_part3



With normalization:



python -m scripts.run\_part3 --normalize



Force number of clusters:



python -m scripts.run\_part3 --k\_override 6



Generate PCA visualization:



python -m scripts.run\_part3 --plot pca



Generate UMAP visualization (requires umap-learn):



pip install umap-learn

python -m scripts.run\_part3 --plot umap



Outputs saved in:

outputs/part3/



============================================================

DATASET



20 Newsgroups Dataset



~18,000 documents



20 balanced classes



Multi-class classification task



Automatically downloaded using sklearn



============================================================

NOTES



All outputs are stored under the outputs/ directory.



Embeddings are cached to speed up reruns.



Delete outputs/cache/ to force embedding recomputation.



Part 3 performs:



Elbow search (K = 2 to 9)



Top-level clustering



Second-level clustering on 2 largest clusters



Topic tree generation

