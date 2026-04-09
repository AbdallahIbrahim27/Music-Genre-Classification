"""
predict.py
----------
Predict the genre of a single audio file using a saved model.

Usage:
    # Tabular model
    python predict.py --model tabular_best_model.pkl  --audio song.wav

    # CNN model
    python predict.py --model cnn_best_model.keras    --audio song.wav
"""

import argparse
import numpy as np
import joblib
import os

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']


def predict_tabular(model_path: str, audio_path: str):
    from feature_extraction import extract_features
    bundle = joblib.load(model_path)
    model  = bundle['model']
    le     = bundle['label_encoder']
    fcols  = bundle['feature_cols']

    feats = extract_features(audio_path)
    if feats is None:
        print("Feature extraction failed.")
        return

    X = np.array([[feats.get(c, 0.0) for c in fcols]])
    pred = model.predict(X)[0]
    label = le.inverse_transform([pred])[0]

    # Probabilities (works for pipelines with predict_proba)
    try:
        proba = model.predict_proba(X)[0]
        top3 = sorted(zip(le.classes_, proba), key=lambda x: -x[1])[:3]
        print(f"\n🎵 Predicted genre: {label.upper()}")
        print("Top-3 probabilities:")
        for g, p in top3:
            bar = '█' * int(p * 30)
            print(f"  {g:12s} {p*100:5.1f}%  {bar}")
    except AttributeError:
        print(f"\n🎵 Predicted genre: {label.upper()}")


def predict_cnn(model_path: str, audio_path: str):
    import tensorflow as tf
    from spectrogram_gen import wav_to_melspec

    model = tf.keras.models.load_model(model_path)
    img = wav_to_melspec(audio_path)
    if img is None:
        print("Spectrogram generation failed.")
        return

    x = img.astype('float32') / 255.0
    x = np.expand_dims(x, axis=0)
    proba = model.predict(x, verbose=0)[0]
    pred  = int(np.argmax(proba))
    label = GENRES[pred]

    print(f"\n🎵 Predicted genre: {label.upper()}")
    print("Top-3 probabilities:")
    top3 = sorted(enumerate(proba), key=lambda x: -x[1])[:3]
    for idx, p in top3:
        bar = '█' * int(p * 30)
        print(f"  {GENRES[idx]:12s} {p*100:5.1f}%  {bar}")


def main():
    parser = argparse.ArgumentParser(description='Predict music genre from audio file.')
    parser.add_argument('--model', required=True, help='Path to saved model (.pkl or .keras)')
    parser.add_argument('--audio', required=True, help='Path to audio file (.wav)')
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"Audio file not found: {args.audio}")
        return

    if args.model.endswith('.pkl'):
        predict_tabular(args.model, args.audio)
    elif args.model.endswith('.keras') or args.model.endswith('.h5'):
        predict_cnn(args.model, args.audio)
    else:
        print("Unknown model format. Use .pkl (tabular) or .keras (CNN).")


if __name__ == '__main__':
    main()
