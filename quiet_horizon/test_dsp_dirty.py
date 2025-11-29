import os
from quiet_horizon_dsp import QuietHorizonDSP  # adjust import if needed

DATASET_DIR = "quiet_horizon/dsp_dataset"

def iter_audio_files(root: str):
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith((".wav", ".flac", ".mp3", ".ogg")):
                yield os.path.join(dirpath, name)

def main():
    dsp = QuietHorizonDSP()

    total = 0
    correct = 0

    # (category, purity) -> stats dict
    bucket_stats: dict[tuple[str, str], dict] = {}

    for path in iter_audio_files(DATASET_DIR):
        rel = os.path.relpath(path, DATASET_DIR)
        parts = rel.split(os.sep)

        if len(parts) < 3:
            category = "unknown"
            purity = "unknown"
        else:
            category = parts[0]
            purity = parts[1]

        expected_anthro = True  # all these are non-animal noise

        result = dsp.analyze(path)
        predicted_anthro = result["anthropogenic"]

        freq = result["frequency"]
        flat = result["flatness"]
        rhythm = result["rhythm"]

        bands = freq.get("bands", {})
        low_frac = bands.get("low_frac", 0.0)
        high_frac = bands.get("high_frac", 0.0)

        mean_flatness = flat.get("mean_flatness", 0.0)
        high_flat_frame_frac = flat.get("high_flat_frame_frac", 0.0)

        rhythmic = bool(rhythm.get("rhythmic", False))
        num_peaks = len(rhythm.get("peaks", []))

        total += 1
        if expected_anthro == predicted_anthro:
            correct += 1

        key = (category, purity)
        if key not in bucket_stats:
            bucket_stats[key] = {
                "total": 0,
                "correct": 0,
                "sum_low_frac": 0.0,
                "sum_high_frac": 0.0,
                "sum_mean_flatness": 0.0,
                "sum_high_flat_frame_frac": 0.0,
                "sum_num_peaks": 0,
                "rhythmic_count": 0,
            }

        stats = bucket_stats[key]
        stats["total"] += 1
        if expected_anthro == predicted_anthro:
            stats["correct"] += 1

        stats["sum_low_frac"] += low_frac
        stats["sum_high_frac"] += high_frac
        stats["sum_mean_flatness"] += mean_flatness
        stats["sum_high_flat_frame_frac"] += high_flat_frame_frac
        stats["sum_num_peaks"] += num_peaks
        if rhythmic:
            stats["rhythmic_count"] += 1

    # --- Overall summary ---
    print(f"\n=== Overall DSP Performance on DIRTY clips ===")
    acc = correct / total if total > 0 else 0.0
    print(f"Total files: {total}")
    print(f"Correct anthropogenic detections: {correct}")
    print(f"Accuracy: {acc:.3f}")

    # --- Per-bucket summary with filter stats ---
    print("\n=== Per-bucket stats (category/purity) ===")
    for (category, purity), s in sorted(bucket_stats.items()):
        t = s["total"]
        c = s["correct"]
        bucket_acc = c / t if t > 0 else 0.0

        avg_low = s["sum_low_frac"] / t if t else 0.0
        avg_high = s["sum_high_frac"] / t if t else 0.0
        avg_flat = s["sum_mean_flatness"] / t if t else 0.0
        avg_high_flat = s["sum_high_flat_frame_frac"] / t if t else 0.0
        avg_peaks = s["sum_num_peaks"] / t if t else 0.0
        frac_rhythmic = s["rhythmic_count"] / t if t else 0.0

        print(
            f"{category}/{purity}: {c}/{t} correct ({bucket_acc:.3f}) | "
            f"low={avg_low:.3f}, high={avg_high:.3f}, "
            f"flat={avg_flat:.3f}, flat_hi_frac={avg_high_flat:.3f}, "
            f"peaks={avg_peaks:.2f}, rhythmic%={frac_rhythmic:.2f}"
        )

if __name__ == "__main__":
    main()
