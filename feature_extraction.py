"""
feature_extraction.py
---------------------
Extracts MFCC and other audio features from audio files using Librosa.
Used by the tabular approach.
"""

import os
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

SAMPLE_RATE = 22050
DURATION    = 30          # seconds per clip
N_MFCC      = 13


def extract_features(file_path: str, sr: int = SAMPLE_RATE) -> dict | None:
    """
    Load an audio file and return a flat dictionary of features:
      - 13 MFCC means + 13 MFCC stds
      - chroma mean/std
      - spectral centroid mean/std
      - spectral rolloff mean/std
      - zero-crossing rate mean/std
      - RMS energy mean/std
    Returns None on failure.
    """
    try:
        y, sr = librosa.load(file_path, sr=sr, duration=DURATION, mono=True)
    except Exception as e:
        print(f"  [WARN] Cannot load {file_path}: {e}")
        return None

    features = {}

    # --- MFCCs ---
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        features[f'mfcc_{i+1}_mean'] = float(np.mean(mfcc[i]))
        features[f'mfcc_{i+1}_std']  = float(np.std(mfcc[i]))

    # --- Chroma ---
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features['chroma_mean'] = float(np.mean(chroma))
    features['chroma_std']  = float(np.std(chroma))

    # --- Spectral centroid ---
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)
    features['spectral_centroid_mean'] = float(np.mean(sc))
    features['spectral_centroid_std']  = float(np.std(sc))

    # --- Spectral rolloff ---
    ro = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features['spectral_rolloff_mean'] = float(np.mean(ro))
    features['spectral_rolloff_std']  = float(np.std(ro))

    # --- Zero-crossing rate ---
    zcr = librosa.feature.zero_crossing_rate(y)
    features['zcr_mean'] = float(np.mean(zcr))
    features['zcr_std']  = float(np.std(zcr))

    # --- RMS energy ---
    rms = librosa.feature.rms(y=y)
    features['rms_mean'] = float(np.mean(rms))
    features['rms_std']  = float(np.std(rms))

    return features


def build_feature_dataframe(data_root: str, csv_out: str = 'features.csv') -> pd.DataFrame:
    """
    Walk `data_root/genres_original/<genre>/` directories,
    extract features for every .wav file, and return a DataFrame.
    Also saves to `csv_out`.
    """
    genres_dir = os.path.join(data_root, 'genres_original')
    if not os.path.isdir(genres_dir):
        raise FileNotFoundError(
            f"Expected directory: {genres_dir}\n"
            "Make sure you downloaded and extracted the GTZAN dataset."
        )

    rows = []
    for genre in GENRES:
        genre_path = os.path.join(genres_dir, genre)
        if not os.path.isdir(genre_path):
            print(f"[WARN] Genre folder not found: {genre_path}")
            continue

        files = [f for f in os.listdir(genre_path) if f.endswith('.wav')]
        print(f"Processing {genre:12s} ({len(files)} files)...")

        for fname in tqdm(files, leave=False):
            fpath = os.path.join(genre_path, fname)
            feats = extract_features(fpath)
            if feats is not None:
                feats['genre'] = genre
                feats['filename'] = fname
                rows.append(feats)

    df = pd.DataFrame(rows)
    df.to_csv(csv_out, index=False)
    print(f"\n✅ Features saved to {csv_out}  ({len(df)} rows, {len(df.columns)} cols)")
    return df


if __name__ == '__main__':
    df = build_feature_dataframe('data', csv_out='features.csv')
    print(df.head())
