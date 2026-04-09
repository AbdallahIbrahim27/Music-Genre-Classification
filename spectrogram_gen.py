"""
spectrogram_gen.py
------------------
Converts each .wav file into a Mel-spectrogram image (PNG).
Images are saved to `images/<genre>/` for use by the CNN.
"""

import os
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')          # non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm

GENRES      = ['blues', 'classical', 'country', 'disco', 'hiphop',
               'jazz', 'metal', 'pop', 'reggae', 'rock']
SAMPLE_RATE = 22050
DURATION    = 30
IMG_SIZE    = 128              # pixels (square output)
N_MELS      = 128
HOP_LENGTH  = 512


def wav_to_melspec(file_path: str, sr: int = SAMPLE_RATE,
                   n_mels: int = N_MELS, duration: float = DURATION,
                   img_size: int = IMG_SIZE) -> np.ndarray | None:
    """
    Load audio → compute log-Mel spectrogram → return as (img_size, img_size, 3) uint8.
    Returns None on failure.
    """
    try:
        y, sr = librosa.load(file_path, sr=sr, duration=duration, mono=True)
    except Exception as e:
        print(f"  [WARN] {file_path}: {e}")
        return None

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)   # log scale

    # Render to RGB image via matplotlib
    fig, ax = plt.subplots(figsize=(img_size / 100, img_size / 100), dpi=100)
    librosa.display.specshow(mel_db, sr=sr, hop_length=HOP_LENGTH,
                              x_axis=None, y_axis=None, ax=ax, cmap='inferno')
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Grab pixel buffer (tostring_rgb removed in matplotlib >= 3.8)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img = buf.reshape(h, w, 4)[:, :, :3]   # RGBA → RGB
    plt.close(fig)

    # Resize to IMG_SIZE × IMG_SIZE
    from PIL import Image
    pil = Image.fromarray(img).resize((img_size, img_size), Image.LANCZOS)
    return np.array(pil)


def generate_spectrograms(data_root: str, out_root: str = 'images') -> None:
    """
    Iterate all genres, convert .wav → spectrogram PNG, save under `out_root`.
    """
    genres_dir = os.path.join(data_root, 'genres_original')
    os.makedirs(out_root, exist_ok=True)

    for genre in GENRES:
        genre_in  = os.path.join(genres_dir, genre)
        genre_out = os.path.join(out_root,   genre)
        if not os.path.isdir(genre_in):
            print(f"[WARN] Missing: {genre_in}")
            continue

        os.makedirs(genre_out, exist_ok=True)
        files = [f for f in os.listdir(genre_in) if f.endswith('.wav')]
        print(f"  {genre:12s}: {len(files)} files")

        for fname in tqdm(files, desc=genre, leave=False):
            out_name = os.path.splitext(fname)[0] + '.png'
            out_path = os.path.join(genre_out, out_name)
            if os.path.exists(out_path):
                continue          # skip already processed

            img = wav_to_melspec(os.path.join(genre_in, fname))
            if img is not None:
                from PIL import Image
                Image.fromarray(img).save(out_path)

    print(f"\n✅ Spectrograms saved to '{out_root}/'")


if __name__ == '__main__':
    generate_spectrograms('data', out_root='images')