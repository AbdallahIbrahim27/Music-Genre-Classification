# 🎵 Music Genre Classification — Task 6

A complete solution using the GTZAN dataset with **two approaches**:
1. **Tabular approach** — MFCCs + Scikit-learn (Random Forest, SVM)
2. **Image-based approach** — Mel spectrograms + CNN (Keras)

## Setup

```bash
pip install librosa scikit-learn tensorflow keras matplotlib numpy pandas seaborn kaggle
```

## Dataset
Download GTZAN from Kaggle:
```bash
kaggle datasets download -d andradaolteanu/gtzan-dataset-music-genre-classification
unzip gtzan-dataset-music-genre-classification.zip -d data/
```

## Run

```bash
# Tabular approach
python tabular_approach.py

# Image-based CNN approach
python cnn_approach.py

# Compare both
python compare_results.py
```
