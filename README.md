# Topic-Enhanced LightGCN (TE-LGCN)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

A novel recommendation system that enhances LightGCN with semantic content features through a dual enhancement strategy combining Doc2Vec embeddings and LDA topic modeling.

## 🎯 Overview

**TE-LGCN (Topic-Enhanced LightGCN)** addresses the cold-start problem and improves recommendation quality by incorporating content-based semantic information into graph collaborative filtering. Unlike vanilla LightGCN which relies solely on user-item interactions, TE-LGCN leverages:

- **Semantic Initialization**: Doc2Vec embeddings from item content (e.g., movie plot summaries)
- **Structural Expansion**: LDA-extracted topics as bridge nodes in the user-item graph
- **Content Consistency**: A novel loss function that preserves semantic meaning during training

### Key Results

| Model | Recall@10 | Improvement |
|-------|-----------|-------------|
| Baseline LightGCN | ~0.159 | - |
| **TE-LGCN (k=10)** | **~0.200** | **+26.4%** |

---

## 🔑 Key Features

### 1. Dual Enhancement Strategy

#### Semantic Initialization (Doc2Vec)
- Pre-trained document embeddings initialize item representations
- Captures semantic similarity from textual content
- Reduces cold-start issues for new items

#### Structural Expansion (LDA Topics)
- Topic nodes create semantic bridges in the graph
- Heterogeneous graph: User-Item-Topic connections
- Enables recommendation via shared topic preferences

### 2. Content Consistency Loss

Combines three objectives:
```
L_total = L_BPR + λ₁ · L_reg + λ₂ · L_content

where:
- L_BPR: Bayesian Personalized Ranking loss
- L_reg: L2 regularization on embeddings
- L_content: L2 distance between learned and fixed Doc2Vec embeddings
```

### 3. Modular Python Package

Import and use TE-LGCN components in your own code:
```python
from te_lgcn.models import TELightGCN
from te_lgcn.training import Trainer
from te_lgcn.evaluation import evaluate_model
```

---

## 📦 Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/topic-enhanced-lightgcn.git
cd topic-enhanced-lightgcn

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode (recommended for development)
pip install -e .
```

### Dependencies

Core libraries:
- `torch>=2.0.0` - Deep learning framework
- `gensim>=4.0.0` - Doc2Vec and LDA implementation
- `pandas`, `numpy`, `scipy` - Data processing
- `scikit-learn` - Additional ML utilities

---

## 🚀 Quick Start

### Option 1: Using Jupyter Notebooks (Recommended for Research)

Follow the execution sequence in [`notebooks/README.md`](notebooks/README.md):

1. **Data Preprocessing**: `notebooks/preprocessing/data_preparation.ipynb`
2. **Feature Extraction**:
   - `notebooks/feature_extraction/doc2vec_embeddings.ipynb`
   - `notebooks/feature_extraction/lda_topics.ipynb`
3. **Baseline**: `notebooks/baselines/lightgcn_baseline.ipynb`
4. **TE-LGCN**: `notebooks/te_lgcn/te_lgcn_k10.ipynb`

### Option 2: Using Python Package

```python
import torch
from te_lgcn.models import TELightGCN
from te_lgcn.training import Trainer
from te_lgcn.evaluation import evaluate_model

# Load your data
# ... (see notebooks for data loading examples)

# Create model
model = TELightGCN(
    n_users=670,
    n_items=3485,
    n_topics=10,              # Number of LDA topics
    dim=64,                   # Embedding dimension
    layers=3,                 # GCN layers
    A_hat=adj_matrix,         # Normalized adjacency matrix
    doc2vec_weights=doc2vec_emb  # Pre-trained Doc2Vec embeddings
).to(device)

# Setup training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    device='cuda',
    lambda1=1e-5,  # L2 regularization weight
    lambda2=1e-3   # Content consistency weight
)

# Train
for epoch in range(50):
    loss = trainer.train_epoch(train_loader)
    results = evaluate_model(model, val_df, user_pos_items, k=10)
    print(f"Epoch {epoch}: Loss={loss:.4f}, Recall@10={results['Recall']:.4f}")
```

---

## 📁 Repository Structure

```
topic-enhanced-lightgcn/
├── te_lgcn/                    # Python package (importable)
│   ├── models/                 # LightGCN and TE-LGCN implementations
│   ├── data/                   # Dataset, graph construction
│   ├── features/               # Doc2Vec and LDA extractors
│   ├── training/               # Trainer and loss functions
│   ├── evaluation/             # Metrics (Recall, NDCG, Precision)
│   └── utils/                  # Configuration and logging
│
├── notebooks/                  # Jupyter notebooks organized by research phase
│   ├── preprocessing/          # Data preparation and filtering
│   ├── feature_extraction/     # Doc2Vec and LDA feature generation
│   ├── baselines/              # Baseline LightGCN experiments
│   └── te_lgcn/                # TE-LGCN experiments (k=7,10,15,20)
│
├── configs/                    # YAML configuration files
│   ├── default.yaml            # Default hyperparameters
│   └── experiments/            # Experiment-specific configs
│
├── data/                       # Data directory (not included in repo)
│   ├── raw/                    # Original datasets
│   ├── processed/              # Filtered and split data
│   └── embeddings/             # Doc2Vec embeddings
│
├── docs/                       # Documentation
│   ├── pipeline.md             # Research methodology
│   └── implementation_summary.md
│
├── results/                    # Experiment results (git-ignored)
├── scripts/                    # CLI scripts (future)
├── tests/                      # Unit tests (future)
│
├── setup.py                    # Package installation
├── requirements.txt            # Dependencies
├── .gitignore                  # Git exclusions
├── LICENSE                     # MIT License
└── README.md                   # This file
```

---

## 📊 Data

This project uses the **MovieLens dataset** with additional movie plot summaries.

### Dataset Statistics (k=5 filtered)

- **Users**: 671
- **Items**: 3,485
- **Interactions**: 89,927
- **Sparsity**: 96.15%

### Download Data

The actual data files are not included in this repository. Please:

1. Download MovieLens dataset from [GroupLens](https://grouplens.org/datasets/movielens/)
2. Place files in `data/raw/`:
   - `ratings.csv`
   - `movies.csv`
   - `movie_data_final_clean.csv` (plot summaries)
3. Run preprocessing notebook to generate processed data

See [`data/README.md`](data/README.md) for detailed data structure and usage.

---

## ⚙️ Hyperparameters

### Default Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `dim` | 64 | Embedding dimension |
| `layers` | 3 | Number of GCN layers |
| `batch_size` | 1024 | Training batch size |
| `lr` | 1e-3 | Learning rate (Adam) |
| `λ₁` | 1e-5 | L2 regularization weight |
| `λ₂` | 1e-3 | Content consistency loss weight |
| `k_core` | 5 | Minimum interactions per user/item |
| `n_topics` | 10 | Number of LDA topics |

### Experiment Configurations

See [`configs/experiments/`](configs/experiments/) for pre-defined configurations:
- `baseline.yaml` - Vanilla LightGCN
- `te_lgcn_k10.yaml` - Full TE-LGCN with 10 topics

---

## 🧪 Experiments

### Baseline Comparison

| Model | Doc2Vec Init | Topic Nodes | Recall@10 | NDCG@10 |
|-------|--------------|-------------|-----------|---------|
| LightGCN | ❌ | ❌ | 0.159 | - |
| LightGCN + Doc2Vec | ✅ | ❌ | ~0.175 | - |
| **TE-LGCN** | ✅ | ✅ | **0.200** | - |

### Ablation Study: Number of Topics

| Topics (k) | Recall@10 | Note |
|------------|-----------|------|
| k=7 | ~0.195 | Fewer topics |
| **k=10** | **~0.200** | **Optimal** |
| k=15 | ~0.198 | More granular |
| k=20 | ~0.196 | Too many topics |

---

## 📈 Evaluation Metrics

Supported metrics (all computed at top-K):
- **Recall@K**: Proportion of relevant items retrieved
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **Precision@K**: Precision of top-K recommendations
- **Hit Rate@K**: Whether any relevant item is in top-K

Default evaluation: K=10

---

## 🛠️ Usage Examples

### Training with Custom Data

```python
from te_lgcn.data import build_heterogeneous_graph, RecommendationDataset
from torch.utils.data import DataLoader

# Build graph with topic nodes
adj_matrix = build_heterogeneous_graph(
    train_df,
    topic_df,
    n_users=n_users,
    n_items=n_items,
    n_topics=n_topics
)

# Create dataset
dataset = RecommendationDataset(train_df, n_items)
train_loader = DataLoader(dataset, batch_size=1024, shuffle=True)

# Train model (see Quick Start section)
```

### Using Pre-trained Embeddings

```python
from te_lgcn.features import Doc2VecExtractor
import pickle

# Option 1: Train Doc2Vec
extractor = Doc2VecExtractor(vector_size=64, epochs=20)
doc2vec_weights = extractor.fit_transform(item_documents)

# Option 2: Load pre-trained embeddings
with open('data/embeddings/doc2vec_embeddings_64d.pkl', 'rb') as f:
    data = pickle.load(f)
    doc2vec_weights = torch.FloatTensor(data['embeddings'])
```

### Extracting Topics with LDA

```python
from te_lgcn.features import LDAExtractor

extractor = LDAExtractor(n_topics=10)
topic_df = extractor.fit_transform(item_documents)
# Returns DataFrame: [movie_id, topic_id, probability]
```

---

## 🔬 Research Methodology

The complete research pipeline is documented in [`docs/pipeline.md`](docs/pipeline.md):

1. **Data Preprocessing**: K-core filtering, leave-one-out splitting
2. **Feature Extraction**: Doc2Vec training, LDA topic modeling
3. **Graph Construction**: Heterogeneous user-item-topic graph
4. **Model Training**: Dual enhancement with content consistency loss
5. **Evaluation**: Multiple metrics on test set

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add unit tests (`tests/`)
- [ ] Implement CLI scripts (`scripts/train.py`, `scripts/evaluate.py`)
- [ ] Support additional datasets (Amazon, Yelp, etc.)
- [ ] Add more baseline models (NGCF, DGCF, etc.)
- [ ] Hyperparameter tuning with Ray Tune
- [ ] Documentation website (Sphinx)

Please open an issue or pull request for any contributions.

---

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@software{te_lgcn_2026,
  title={Topic-Enhanced LightGCN: A Dual Enhancement Strategy for Recommendation},
  author={TE-LGCN Research Team},
  year={2026},
  url={https://github.com/yourusername/topic-enhanced-lightgcn}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **LightGCN**: He et al., "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation", SIGIR 2020
- **MovieLens**: Harper and Konstan, "The MovieLens Datasets", ACM TIIS 2015
- **Gensim**: Řehůřek and Sojka, "Software Framework for Topic Modelling with Large Corpora", 2010

---

## 📞 Contact

For questions or issues:
- **Issues**: [GitHub Issues](https://github.com/yourusername/topic-enhanced-lightgcn/issues)
- **Email**: your.email@example.com

---

**Built with ❤️ for the recommendation systems research community**
