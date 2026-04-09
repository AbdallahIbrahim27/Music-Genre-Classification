# 🎵 Music Genre Classification

A machine learning project that automatically detects the genre of a music track from audio using two different approaches — traditional ML with handcrafted features, and deep learning with image-based spectrograms.

Built on the **GTZAN dataset**, which contains 1,000 audio clips across 10 genres: Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, and Rock.

---

## 🧠 The Problem

Computers don't understand sound the way humans do. To classify music, we first need to convert raw audio into a numerical representation that a model can learn from. This project solves that using **two strategies**:

---

## 🔀 Approach 1 — Tabular (MFCC Features + Scikit-learn)

Instead of feeding the raw audio to the model, we extract a set of **56 numerical features** that summarize the audio mathematically.

### Features extracted per clip:
| Feature | What it captures |
|---|---|
| **MFCCs** (13 coefficients × mean + std) | Timbre and texture — how the sound feels |
| **Chroma** (mean + std) | Harmonic content — the notes and chords |
| **Spectral Centroid** (mean + std) | Brightness — where energy is concentrated |
| **Spectral Rolloff** (mean + std) | The frequency below which 85% of energy lies |
| **Zero-Crossing Rate** (mean + std) | How noisy vs. tonal the signal is |
| **RMS Energy** (mean + std) | Loudness and dynamics |

These 56 numbers are saved as a CSV and fed to two classifiers:

- **Random Forest** — 300 decision trees voting on the genre
- **SVM (RBF kernel)** — finds optimal hyperplanes in high-dimensional feature space

**Expected accuracy: ~78–85%**

---

## 🖼️ Approach 2 — Image-based (Mel Spectrogram + CNN)

Instead of summarizing the audio as numbers, we convert each clip into a **visual image** called a Mel Spectrogram.

- The **X-axis** represents time
- The **Y-axis** represents frequency (Mel scale, matching human hearing)
- The **color** represents the intensity (loudness) at each frequency and time

Each genre produces a visually distinct pattern — Metal looks dense and dark, Classical shows clean structured regions, Hip-Hop has strong low-frequency rhythms.

These 128×128 images are fed into a **CNN (Convolutional Neural Network)** — the same type used in facial recognition and image classification — which learns the visual patterns that distinguish each genre.

### Two CNN models:

**A) Custom CNN from scratch**
```
Conv2D(32) → BatchNorm → MaxPool
Conv2D(64) → BatchNorm → MaxPool
Conv2D(128) → BatchNorm → MaxPool
Conv2D(256) → GlobalAveragePool
Dense(512) → Dropout(0.5)
Dense(256) → Dropout(0.3)
Dense(10, softmax)
```

**B) MobileNetV2 (Transfer Learning)** — Bonus
A pre-trained model on ImageNet is adapted for spectrograms by:
1. Freezing the base layers and training only the new classification head
2. Fine-tuning the last 20 layers with a very small learning rate

**Expected accuracy: ~85–92%**

---

## 📊 Comparison

| Model | Type | Expected Accuracy |
|---|---|---|
| Random Forest | Tabular | ~80–83% |
| SVM (RBF) | Tabular | ~78–85% |
| Custom CNN | Image | ~85–90% |
| MobileNetV2 | Image (Transfer) | ~88–92% |

The CNN approach outperforms tabular because spectrograms preserve the **full time-frequency structure** of the audio, while the tabular approach compresses everything into 56 numbers and loses detail.

---

## 🌐 Streamlit Web App

An interactive web application that lets you upload any audio file and get an instant genre prediction with a waveform, Mel spectrogram, and MFCC visualization.

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
Music-Genre-Classification/
│
├── app.py                  # Streamlit web app
├── feature_extraction.py   # Extract MFCCs and audio features
├── tabular_approach.py     # Train Random Forest + SVM
├── cnn_approach.py         # Train Custom CNN + MobileNetV2
├── spectrogram_gen.py      # Convert audio to spectrogram images
├── compare_results.py      # Side-by-side model comparison
├── predict.py              # Predict genre for a single audio file
└── requirements.txt        # Dependencies
```

---

## ⚙️ Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the GTZAN dataset
```bash
kaggle datasets download -d andradaolteanu/gtzan-dataset-music-genre-classification
unzip gtzan-dataset-music-genre-classification.zip -d data/
```

### 3. Train the tabular model
```bash
python tabular_approach.py
# Outputs: tabular_best_model.pkl, features.csv, confusion matrix plots
```

### 4. Train the CNN model
```bash
python cnn_approach.py
# Outputs: cnn_best_model.keras, training curves, confusion matrix
```

### 5. Compare both approaches
```bash
python compare_results.py
# Outputs: final_comparison.png
```

### 6. Predict a single file
```bash
python predict.py --model tabular_best_model.pkl --audio mysong.wav
python predict.py --model cnn_best_model.keras   --audio mysong.wav
```

### 7. Launch the web app
```bash
streamlit run app.py
```

---

## 📦 Dependencies

```
librosa          # Audio loading and feature extraction
scikit-learn     # Random Forest, SVM, preprocessing
tensorflow       # CNN and transfer learning
matplotlib       # Visualization and spectrogram plots
numpy / pandas   # Data handling
streamlit        # Web app
Pillow           # Image processing
```

---

## 📚 Dataset

**GTZAN Genre Collection** — Tzanetakis & Cook (2002)
- 1,000 audio tracks, 30 seconds each
- 10 genres × 100 tracks
- 22,050 Hz sample rate, mono

[Download from Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

---

## 🗂️ Topics

`machine-learning` `deep-learning` `audio-classification` `librosa` `cnn` `streamlit` `gtzan` `music-genre` `mel-spectrogram` `mfcc` `transfer-learning` `scikit-learn`
