import os
import numpy as np
import librosa
import matplotlib.pyplot as plt

# -------- CONFIG --------
SRC_ROOT = "quiet_horizon/dataset_cnn"        # anthro / nature audio
DST_ROOT = "quiet_horizon/dataset_cnn_specs"  # anthro / nature spectrogram PNGs

SR = 22050
DURATION = 2.5       # seconds (clips will be padded/cropped to this)
N_MELS = 128
HOP_LENGTH = 512


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def audio_to_fixed_length(y: np.ndarray, sr: int, duration: float) -> np.ndarray:
    target_len = int(sr * duration)
    if len(y) < target_len:
        pad = target_len - len(y)
        y = np.pad(y, (0, pad))
    elif len(y) > target_len:
        start = (len(y) - target_len) // 2
        y = y[start:start + target_len]
    return y


def make_mel_spectrogram(y: np.ndarray, sr: int) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
        power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db


def save_spectrogram_image(S_db: np.ndarray, out_path: str):
    # 128x128-ish image
    plt.figure(figsize=(2.56, 2.56), dpi=50)
    plt.axis("off")
    plt.imshow(S_db, origin="lower", aspect="auto", cmap="magma")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def process_split(split_root: str, split_out_root: str):
    """
    split_root: SRC_ROOT/anthro or SRC_ROOT/nature
    split_out_root: DST_ROOT/anthro or DST_ROOT/nature
    """
    ensure_dir(split_out_root)

    count = 0
    for dirpath, _, filenames in os.walk(split_root):
        for name in filenames:
            if not name.lower().endswith((".wav", ".flac", ".mp3", ".ogg")):
                continue

            src_path = os.path.join(dirpath, name)

            # preserve relative subfolder structure (if any)
            rel_dir = os.path.relpath(dirpath, split_root)
            out_dir = os.path.join(split_out_root, rel_dir)
            ensure_dir(out_dir)

            stem, _ = os.path.splitext(name)
            out_path = os.path.join(out_dir, f"{stem}.png")

            if os.path.exists(out_path):
                continue  # skip if already generated

            try:
                y, sr = librosa.load(src_path, sr=SR, mono=True)
                y = audio_to_fixed_length(y, sr, DURATION)
                S_db = make_mel_spectrogram(y, sr)
                save_spectrogram_image(S_db, out_path)
                count += 1
                if count % 200 == 0:
                    print(f"{split_out_root}: generated {count} images...")
            except Exception as e:
                print(f"[WARN] Failed on {src_path}: {e}")

    print(f"Done: {split_out_root} total new images: {count}")


def main():
    anthro_src = os.path.join(SRC_ROOT, "anthro")
    nature_src = os.path.join(SRC_ROOT, "nature")

    anthro_out = os.path.join(DST_ROOT, "anthro")
    nature_out = os.path.join(DST_ROOT, "nature")

    print("Generating spectrograms for anthro...")
    process_split(anthro_src, anthro_out)

    print("Generating spectrograms for nature...")
    process_split(nature_src, nature_out)

    print("All done.")


if __name__ == "__main__":
    main()
