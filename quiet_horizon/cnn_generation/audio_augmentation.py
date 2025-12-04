import os
import random
import uuid

import numpy as np
import librosa
import soundfile as sf

DATASET_ROOT = "quiet_horizon/dataset_cnn"
ANTHRO_DIR = os.path.join(DATASET_ROOT, "anthro")
NATURE_DIR = os.path.join(DATASET_ROOT, "nature")

ANTHRO_TARGET_RATIO = 0.7
MAX_AUG_PER_FILE = 3
SR = 22050


def list_audio_files(root):
    exts = (".wav", ".flac", ".ogg", ".mp3")
    files = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(exts):
                files.append(os.path.join(dirpath, name))
    return files


def random_gain(y):
    factor = random.uniform(0.7, 1.3)
    y2 = y * factor
    return np.clip(y2, -1.0, 1.0)


def random_time_stretch(y):
    """
    Simple time-stretch using interpolation instead of librosa.effects.time_stretch
    to avoid any name-shadowing issues.
    """
    rate = random.uniform(0.9, 1.1)  # < 1 = slower/longer, > 1 = faster/shorter

    orig_len = len(y)
    if orig_len < 2:
        return y

    # new length scaled by 1/rate so that time is stretched or compressed
    new_len = max(2, int(round(orig_len / rate)))

    x_old = np.linspace(0.0, 1.0, orig_len)
    x_new = np.linspace(0.0, 1.0, new_len)

    y_stretched = np.interp(x_new, x_old, y)

    # pad or crop back to original length
    if len(y_stretched) > orig_len:
        y_stretched = y_stretched[:orig_len]
    else:
        pad = orig_len - len(y_stretched)
        y_stretched = np.pad(y_stretched, (0, pad))

    return y_stretched


def random_pitch_shift(y, sr):
    steps = random.uniform(-2.0, 2.0)
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)


def add_noise(y):
    noise_level = random.uniform(0.005, 0.02)
    noise = np.random.randn(len(y)) * noise_level
    y2 = y + noise
    return np.clip(y2, -1.0, 1.0)


def overlay_nature(y_anthro, sr, nature_files, mix_level=0.25):
    """
    Mix a random nature clip into the anthro clip at a lower level.
    mix_level ~ 0.25 => nature at 25% amplitude relative to anthro.
    """
    if not nature_files:
        return y_anthro

    nat_path = random.choice(nature_files)
    y_nat, sr_nat = librosa.load(nat_path, sr=sr, mono=True)

    # Match length
    if len(y_nat) > len(y_anthro):
        y_nat = y_nat[:len(y_anthro)]
    else:
        pad = len(y_anthro) - len(y_nat)
        y_nat = np.pad(y_nat, (0, pad))

    # Simple normalization/safety
    max_abs = np.max(np.abs(y_nat))
    if max_abs > 0:
        y_nat = y_nat / max_abs * 0.8

    y_mix = y_anthro + mix_level * y_nat
    return np.clip(y_mix, -1.0, 1.0)


def random_augment(y, sr, nature_files):
    """
    Apply 1–3 random augmentations.
    Nature overlay is one possible augmentation in the chain.
    """
    aug_fns = [
        lambda z: random_gain(z),
        lambda z: random_time_stretch(z),
        lambda z: random_pitch_shift(z, sr),
        lambda z: add_noise(z),
        lambda z: overlay_nature(z, sr, nature_files),
    ]

    k = random.randint(1, 3)
    fns = random.sample(aug_fns, k=k)

    y2 = y.copy()
    for fn in fns:
        y2 = fn(y2)
    return y2


def main():
    nature_files = list_audio_files(NATURE_DIR)
    anthro_files = list_audio_files(ANTHRO_DIR)

    n_nature = len(nature_files)
    n_anthro = len(anthro_files)

    print(f"Nature clips: {n_nature}")
    print(f"Anthro clips: {n_anthro}")

    target_anthro = int(n_nature * ANTHRO_TARGET_RATIO)
    if n_anthro >= target_anthro:
        print(f"No augmentation needed: anthro ({n_anthro}) >= target ({target_anthro}).")
        return

    needed = target_anthro - n_anthro
    print(f"Target anthro count: {target_anthro}")
    print(f"Need to create ~{needed} augmented anthro clips.")

    aug_counts = {f: 0 for f in anthro_files}
    created = 0

    while created < needed:
        base = random.choice(anthro_files)
        if aug_counts[base] >= MAX_AUG_PER_FILE:
            continue

        y, sr = librosa.load(base, sr=SR, mono=True)
        y_aug = random_augment(y, sr, nature_files)

        folder = os.path.dirname(base)
        base_name = os.path.basename(base)
        stem, ext = os.path.splitext(base_name)
        out_name = f"{stem}_aug_{uuid.uuid4().hex[:8]}{ext}"
        out_path = os.path.join(folder, out_name)

        sf.write(out_path, y_aug, sr)
        aug_counts[base] += 1
        created += 1

        if created % 100 == 0:
            print(f"  Created {created}/{needed} augmented files...")

    print(f"\nDone. Created {created} augmented anthro clips.")
    final_anthro = len(list_audio_files(ANTHRO_DIR))
    print(f"Final anthro count (approx): {final_anthro}")


if __name__ == "__main__":
    main()
