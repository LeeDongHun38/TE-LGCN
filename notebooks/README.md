# Notebooks Execution Guide

This directory contains Jupyter notebooks for the TE-LGCN research pipeline.

## Execution Order

Follow this sequence to reproduce the research:

### 1. Data Preprocessing
**Notebook**: [preprocessing/data_preparation.ipynb](preprocessing/data_preparation.ipynb)

**Purpose**:
- Apply k-core filtering to ensure graph connectivity
- Convert ratings to implicit feedback (≥4.0 = positive)
- Split data into train/val/test sets using leave-one-out strategy
- Create ID mappings (user2idx, item2idx)

**Outputs**:
- `data/processed/k5_filtered/train.csv`
- `data/processed/k5_filtered/val.csv`
- `data/processed/k5_filtered/test.csv`
- `data/processed/k5_filtered/user2idx.pkl`
- `data/processed/k5_filtered/item2idx.pkl`

---

### 2. Feature Extraction

#### 2a. Doc2Vec Embeddings
**Notebook**: [feature_extraction/doc2vec_embeddings.ipynb](feature_extraction/doc2vec_embeddings.ipynb)

**Purpose**:
- Train Doc2Vec on movie plot summaries
- Generate 64-dimensional semantic embeddings
- Save embeddings for item initialization

**Outputs**:
- `data/embeddings/doc2vec_embeddings_64d.pkl`

#### 2b. LDA Topic Modeling
**Notebook**: [feature_extraction/lda_topics.ipynb](feature_extraction/lda_topics.ipynb)

**Purpose**:
- Extract topics from movie plots using LDA
- Create item-topic associations
- Generate different topic variants (k=7, 10, 15, 20)

**Outputs**:
- `data/processed/topic_vectors_7.csv`
- `data/processed/topic_vectors_10.csv`
- `data/processed/topic_vectors_15.csv`
- `data/processed/topic_vectors_20.csv`

---

### 3. Baseline Experiments

**Notebook**: [baselines/lightgcn_baseline.ipynb](baselines/lightgcn_baseline.ipynb)

**Purpose**:
- Train vanilla LightGCN (no Doc2Vec, no topics)
- Establish baseline performance
- Test different regularization parameters

**Results**:
- Baseline Recall@10, NDCG@10, Precision@10

---

### 4. TE-LGCN Experiments

**Notebooks**:
- [te_lgcn/te_lgcn_k7.ipynb](te_lgcn/te_lgcn_k7.ipynb) - 7 topics
- [te_lgcn/te_lgcn_k10.ipynb](te_lgcn/te_lgcn_k10.ipynb) - 10 topics (recommended)
- [te_lgcn/te_lgcn_k15.ipynb](te_lgcn/te_lgcn_k15.ipynb) - 15 topics
- [te_lgcn/te_lgcn_k20.ipynb](te_lgcn/te_lgcn_k20.ipynb) - 20 topics

**Purpose**:
- Train TE-LGCN with full dual enhancement strategy
- Doc2Vec semantic initialization + LDA topic nodes
- Compare different numbers of topics

**Results**:
- TE-LGCN performance vs baseline
- Impact of topic count on recommendation quality

---

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| dim | 64 | Embedding dimension |
| layers | 3 | GCN layers |
| batch_size | 1024 | Training batch size |
| lr | 1e-3 | Learning rate |
| λ1 | 1e-5 | L2 regularization weight |
| λ2 | 1e-3 | Content consistency loss weight |
| k-core | 5 or 10 | Minimum interactions per user/item |

---

## Expected Results

Based on the research pipeline, TE-LGCN should achieve:

| Model | Recall@10 | Improvement |
|-------|-----------|-------------|
| Baseline LightGCN | ~0.159 | - |
| TE-LGCN (k=10) | ~0.200 | +26.4% |

---

## Notes

- All notebooks use Google Colab paths by default (`/content/drive/MyDrive/...`)
- Modify `base_path` variable to run locally
- Ensure Doc2Vec embeddings are generated before running TE-LGCN notebooks
- For ablation studies, you can disable Doc2Vec (`doc2vec_weights=None`) or content loss (`lambda2=0`)

---

## Troubleshooting

**Issue**: "FileNotFoundError: doc2vec_embeddings_64d.pkl"
**Solution**: Run [feature_extraction/doc2vec_embeddings.ipynb](feature_extraction/doc2vec_embeddings.ipynb) first

**Issue**: "Item count mismatch"
**Solution**: Ensure all notebooks use the same data split (k5_filtered or k10_filtered)

**Issue**: Performance worse than baseline
**Solution**: Try different λ2 values (1e-4, 1e-3, 1e-2) or check if Doc2Vec loaded correctly
