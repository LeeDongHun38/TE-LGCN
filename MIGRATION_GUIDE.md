# Migration Guide: Option 3 (Modular Hybrid Structure)

✅ **Migration Complete!** Your TE-LGCN project has been reorganized into a professional GitHub-ready structure.

---

## 📁 New Structure Overview

```
topic-enhanced-lightgcn/
├── te_lgcn/                    # 🆕 Python package (importable)
│   ├── models/                 # LightGCN and TE-LGCN classes
│   ├── data/                   # Dataset, graph construction
│   ├── features/               # Doc2Vec and LDA extractors
│   ├── training/               # Trainer and loss functions
│   ├── evaluation/             # Metrics (Recall, NDCG, etc.)
│   └── utils/                  # Config and logging utilities
│
├── notebooks/                  # 🆕 Organized by research phase
│   ├── preprocessing/          # data_preparation.ipynb
│   ├── feature_extraction/     # doc2vec_embeddings.ipynb, lda_topics.ipynb
│   ├── baselines/              # lightgcn_baseline.ipynb
│   └── te_lgcn/                # te_lgcn_k7/10/15/20.ipynb
│
├── configs/                    # 🆕 YAML configuration files
│   ├── default.yaml
│   └── experiments/
│       ├── baseline.yaml
│       └── te_lgcn_k10.yaml
│
├── data/                       # Data directories with READMEs
├── docs/                       # Documentation (pipeline.md, etc.)
├── results/                    # Experiment results (git-ignored)
├── scripts/                    # CLI scripts (future)
├── tests/                      # Unit tests (future)
│
├── setup.py                    # 🆕 Package installation
├── requirements.txt            # 🆕 Dependencies
├── .gitignore                  # 🆕 Git exclusions
└── LICENSE                     # 🆕 MIT License
```

---

## 🔄 File Mapping (Old → New)

| Original File | New Location |
|--------------|-------------|
| `data_split.ipynb` | `notebooks/preprocessing/data_preparation.ipynb` |
| `Doc2vec.ipynb` | `notebooks/feature_extraction/doc2vec_embeddings.ipynb` |
| `LDA.ipynb` | `notebooks/feature_extraction/lda_topics.ipynb` |
| `BaseLightGCN.ipynb` | `notebooks/baselines/lightgcn_baseline.ipynb` |
| `LDA_LightGCN_7.ipynb` | `notebooks/te_lgcn/te_lgcn_k7.ipynb` |
| `LDA_LightGCN_10.ipynb` | `notebooks/te_lgcn/te_lgcn_k10.ipynb` |
| `LDA_LightGCN_15.ipynb` | `notebooks/te_lgcn/te_lgcn_k15.ipynb` |
| `LDA_LightGCN_20.ipynb` | `notebooks/te_lgcn/te_lgcn_k20.ipynb` |
| `pipeline.md` | `docs/pipeline.md` |
| `IMPLEMENTATION_SUMMARY.md` | `docs/implementation_summary.md` |

**Note**: Original files are still in the root directory. You can safely delete them after verifying the new structure.

---

## 🚀 Quick Start

### 1. Install the Package

```bash
# Install in editable mode (recommended for development)
pip install -e .

# Or install normally
pip install .
```

This makes the `te_lgcn` package importable from anywhere:

```python
from te_lgcn.models import TELightGCN
from te_lgcn.training import Trainer
from te_lgcn.evaluation import evaluate_model
```

### 2. Run Notebooks

Follow the execution guide: [notebooks/README.md](notebooks/README.md)

**Order**:
1. `notebooks/preprocessing/data_preparation.ipynb` - Prepare data
2. `notebooks/feature_extraction/doc2vec_embeddings.ipynb` - Generate embeddings
3. `notebooks/feature_extraction/lda_topics.ipynb` - Extract topics
4. `notebooks/baselines/lightgcn_baseline.ipynb` - Baseline results
5. `notebooks/te_lgcn/te_lgcn_k10.ipynb` - TE-LGCN experiments

### 3. Using the Python Package

```python
import torch
from te_lgcn.models import TELightGCN
from te_lgcn.training import Trainer
from te_lgcn.evaluation import evaluate_model

# Load data and embeddings
# ... (see notebooks for examples)

# Create model
model = TELightGCN(
    n_users=670,
    n_items=3485,
    n_topics=10,
    dim=64,
    layers=3,
    A_hat=adj_matrix,
    doc2vec_weights=doc2vec_emb
)

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
trainer = Trainer(model, optimizer, device='cuda', lambda1=1e-5, lambda2=1e-3)

for epoch in range(50):
    loss = trainer.train_epoch(train_loader)
    results = evaluate_model(model, val_df, user_pos_items, k=10)
    print(f"Epoch {epoch}: Loss={loss:.4f}, Recall@10={results['Recall']:.4f}")
```

---

## 🧹 Cleanup Recommendations

### Files to Delete (Already Copied to New Locations)

```bash
# These are duplicates - originals are now in notebooks/ or docs/
rm data_split.ipynb
rm Doc2vec.ipynb
rm LDA.ipynb
rm BaseLightGCN.ipynb
rm LDA_LightGCN_7.ipynb
rm LDA_LightGCN_10.ipynb
rm LDA_LightGCN_15.ipynb
rm LDA_LightGCN_20.ipynb
rm pipeline.md
rm IMPLEMENTATION_SUMMARY.md
rm REPOSITORY_STRUCTURE_PROPOSALS.md
```

### Files to Keep

- `requirements.txt`
- `setup.py`
- `.gitignore`
- `LICENSE`
- All directories (notebooks/, te_lgcn/, configs/, data/, docs/)

---

## 📝 Before Pushing to GitHub

### 1. Initialize Git Repository (if not already)

```bash
git init
git add .
git commit -m "Initial commit: TE-LGCN research code"
```

### 2. Create README.md

You mentioned you'll write this yourself. Key sections to include:

```markdown
# Topic-Enhanced LightGCN

## Overview
Brief description of TE-LGCN and research goals

## Installation
pip install -r requirements.txt
pip install -e .

## Quick Start
Link to notebooks/README.md

## Repository Structure
Overview of directories

## Citation
How to cite your work

## License
MIT License
```

### 3. Verify .gitignore

The `.gitignore` file excludes:
- ✅ Data files (*.csv, *.pkl, *.npy)
- ✅ Model checkpoints (*.pt, *.pth)
- ✅ Jupyter checkpoints
- ✅ Python cache
- ✅ Results directory
- ✅ Google Colab paths

### 4. Create GitHub Repository

```bash
# On GitHub, create new repository: topic-enhanced-lightgcn
# Then push:
git remote add origin https://github.com/yourusername/topic-enhanced-lightgcn.git
git branch -M main
git push -u origin main
```

---

## 🎯 Benefits of New Structure

### For Researchers
✅ **Clear execution path** - Numbered notebooks show research flow
✅ **Reproducibility** - Config files document exact hyperparameters
✅ **Easy comparison** - Baseline and variants side-by-side

### For Developers
✅ **Importable package** - `from te_lgcn import ...`
✅ **Modular code** - Reuse models, losses, metrics
✅ **Type hints & docs** - Professional code quality

### For Collaborators
✅ **Organized structure** - Know where to find things
✅ **Documentation** - READMEs guide through the project
✅ **Extensible** - Easy to add new experiments

---

## 🔧 Future Enhancements

### Optional Additions

1. **CLI Scripts** (in `scripts/`)
   ```bash
   python scripts/train.py --config configs/experiments/te_lgcn_k10.yaml
   python scripts/evaluate.py --checkpoint results/best_model.pt
   ```

2. **Unit Tests** (in `tests/`)
   ```bash
   pytest tests/test_models.py
   ```

3. **Documentation Website** (using Sphinx)
   ```bash
   cd docs
   make html
   ```

4. **CI/CD** (GitHub Actions)
   - Automatic testing on push
   - Code quality checks

---

## 📊 Package vs Notebooks

### Use Notebooks When:
- Exploring data
- Running experiments
- Generating visualizations
- Writing research narrative

### Use Package When:
- Need reproducible results
- Running multiple experiments programmatically
- Building on top of TE-LGCN
- Sharing code with others

---

## ❓ FAQ

**Q: Can I still use Google Colab?**
A: Yes! Just upload the entire repository to Google Drive and adjust `base_path` in notebooks.

**Q: Do I need to rewrite my notebooks?**
A: No! The notebooks are already moved and ready to use. You can optionally import from `te_lgcn` package for cleaner code.

**Q: How do I add a new experiment?**
A:
1. Create new config in `configs/experiments/my_experiment.yaml`
2. Create notebook in `notebooks/te_lgcn/my_experiment.ipynb`
3. Use existing models from `te_lgcn.models`

**Q: What if I want a different structure?**
A: The structure is flexible! See [REPOSITORY_STRUCTURE_PROPOSALS.md](REPOSITORY_STRUCTURE_PROPOSALS.md) for alternatives.

---

## 📞 Support

- Issues: Create GitHub issues
- Questions: Check `docs/` folder
- Contributions: Pull requests welcome!

---

**Migration completed successfully! 🎉**

Your code is now ready for GitHub publication and collaboration.
