import os
import csv

import numpy as np

from quiet_horizon_dsp import QuietHorizonDSP  # your wrapper

DATASET_DIR = "quiet_horizon/dsp_dataset"
OUT_CSV = "quiet_horizon_features_multiclass.csv"

# 0=nature, 1=mixed, 2=anthro
CLASS_NAMES = {0: "nature", 1: "mixed", 2: "anthro"}


def iter_audio_files(root: str):
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith((".wav", ".flac", ".mp3", ".ogg")):
                full = os.path.join(dirpath, name)
                rel = os.path.relpath(full, root)
                yield full, rel


def assign_labels(rel_path: str) -> dict:
    """
    Given a relative path like:
      anthro/vehicle_-_road/pure/road_0001.wav
      anthro/music/mixed/music_0003.wav
      nature/american_robin/nature/robin_0123.wav

    Return:
      label_3: 0=nature, 1=mixed, 2=anthro
      label_binary: 0=nature, 1=contaminated (mixed or anthro)
      high_level: 'nature' or 'anthro'
      category: species or anthro subtype
      purity: 'nature' | 'pure' | 'mixed'
    """
    parts = rel_path.split(os.sep)

    # basic sanity
    if len(parts) < 3:
        return {
            "label_3": None,
            "label_binary": None,
            "high_level": "unknown",
            "category": "unknown",
            "purity": "unknown",
        }

    top = parts[0]

    if top == "nature":
        # nature/<species>/nature/file.wav
        category = parts[1]
        purity = parts[2]  # should be "nature"
        label_3 = 0
        label_binary = 0

    elif top == "anthro":
        # anthro/<category>/<pure|mixed>/file.wav
        category = parts[1]
        purity = parts[2]

        if purity == "mixed":
            label_3 = 1   # mixed
        else:
            label_3 = 2   # pure anthro

        label_binary = 1

    else:
        # fall back if we add other top-level dirs later
        category = parts[1]
        purity = parts[2]
        label_3 = None
        label_binary = None

    return {
        "label_3": label_3,
        "label_binary": label_binary,
        "high_level": "nature" if label_3 == 0 else "anthro",
        "category": category,
        "purity": purity,
    }


def main():
    dsp = QuietHorizonDSP()

    # define CSV columns
    fieldnames = [
        "path",
        "label_3",
        "label_binary",
        "high_level",   # 'nature' or 'anthro'
        "category",     # species or anthro subtype
        "purity",       # 'nature' | 'pure' | 'mixed'
        # DSP feature columns:
        "low_frac",
        "high_frac",
        "mean_flatness",
        "high_flat_frame_frac",
        "peak_count",
        "rhythmic",
        "too_short",
    ]

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for full_path, rel_path in iter_audio_files(DATASET_DIR):
            labels = assign_labels(rel_path)
            if labels["label_3"] is None:
                print(f"[SKIP] Unknown label for {rel_path}")
                continue

            result = dsp.analyze(full_path)

            freq = result.get("frequency", {})
            flat = result.get("flatness", {})
            rhythm = result.get("rhythm", {})
            reasons = result.get("reasons", {})

            bands = freq.get("bands", {})
            low_frac = bands.get("low_frac", 0.0)
            high_frac = bands.get("high_frac", 0.0)

            mean_flatness = float(flat.get("mean_flatness", 0.0))
            high_flat_frame_frac = float(flat.get("high_flat_frame_frac", 0.0))

            peak_count = int(rhythm.get("peak_count", len(rhythm.get("peaks", []) or [])))
            rhythmic = bool(rhythm.get("rhythmic", False))

            too_short = bool(reasons.get("too_short", False))

            row = {
                "path": rel_path.replace("\\", "/"),
                "label_3": labels["label_3"],
                "label_binary": labels["label_binary"],
                "high_level": labels["high_level"],
                "category": labels["category"],
                "purity": labels["purity"],
                "low_frac": low_frac,
                "high_frac": high_frac,
                "mean_flatness": mean_flatness,
                "high_flat_frame_frac": high_flat_frame_frac,
                "peak_count": peak_count,
                "rhythmic": int(rhythmic),
                "too_short": int(too_short),
            }

            writer.writerow(row)

    print(f"✅ Wrote features to {OUT_CSV}")


if __name__ == "__main__":
    main()
