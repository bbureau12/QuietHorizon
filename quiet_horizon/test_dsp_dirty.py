import os
from quiet_horizon_dsp import QuietHorizonDSP  # adjust import if needed

DATASET_ROOT = "quiet_horizon/dsp_dataset"

def iter_audio_files(root: str):
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith((".wav", ".flac", ".mp3", ".ogg")):
                yield os.path.join(dirpath, name)

def main():
    dsp = QuietHorizonDSP()

    total = 0
    correct = 0
    bucket_stats: dict[tuple[str, str, str], dict] = {}

    for path in iter_audio_files(DATASET_ROOT):
        rel = os.path.relpath(path, DATASET_ROOT)
        parts = rel.split(os.sep)

        if len(parts) < 2:
            # we expect at least anthro/<cat>/... or nature/<species>/...
            continue

        top = parts[0]  # "anthro" or "nature"

        if top == "anthro":
            # anthro/<category>/<purity>/file.wav
            if len(parts) < 4:
                # anthro/category/purity/file
                continue
            source = "anthro"
            category = parts[1]
            purity = parts[2]
            expected_anthro = True

        elif top == "nature":
            # nature/<species>/file.wav
            source = "nature"
            category = parts[1] if len(parts) >= 2 else "unknown"
            purity = "nature"
            expected_anthro = False

        else:
            # ignore anything else at top level
            continue

        result = dsp.analyze(path)
        predicted_anthro = result["anthropogenic"]

        freq = result["frequency"]
        flat = result["flatness"]
        rhythm = result["rhythm"]
        reasons = result.get("reasons", {})

        bands = freq.get("bands", {})
        low_frac = bands.get("low_frac", 0.0)
        high_frac = bands.get("high_frac", 0.0)

        mean_flatness = flat.get("mean_flatness", 0.0)
        high_flat_frame_frac = flat.get("high_flat_frame_frac", 0.0)

        rhythmic = bool(rhythm.get("rhythmic", False))
        num_peaks = len(rhythm.get("peaks", []))

        too_short = bool(reasons.get("too_short", False))

        total += 1
        is_correct = (expected_anthro == predicted_anthro)
        if is_correct:
            correct += 1

        key = (source, category, purity)
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
                "too_short_count": 0,
            }

        s = bucket_stats[key]
        s["total"] += 1
        if is_correct:
            s["correct"] += 1

        s["sum_low_frac"] += low_frac
        s["sum_high_frac"] += high_frac
        s["sum_mean_flatness"] += mean_flatness
        s["sum_high_flat_frame_frac"] += high_flat_frame_frac
        s["sum_num_peaks"] += num_peaks
        if rhythmic:
            s["rhythmic_count"] += 1
        if too_short:
            s["too_short_count"] += 1

    # --- Overall summary ---
    print(f"\n=== Overall DSP Performance (anthro + nature) ===")
    acc = correct / total if total > 0 else 0.0
    print(f"Total files: {total}")
    print(f"Correct labels (anthro vs natural): {correct}")
    print(f"Accuracy: {acc:.3f}")

    # --- Per-bucket summary ---
    print("\n=== Per-bucket stats (source/category/purity) ===")
    for (source, category, purity), s in sorted(bucket_stats.items()):
        t = s["total"]
        c = s["correct"]
        bucket_acc = c / t if t > 0 else 0.0

        avg_low = s["sum_low_frac"] / t if t else 0.0
        avg_high = s["sum_high_frac"] / t if t else 0.0
        avg_flat = s["sum_mean_flatness"] / t if t else 0.0
        avg_high_flat = s["sum_high_flat_frame_frac"] / t if t else 0.0
        avg_peaks = s["sum_num_peaks"] / t if t else 0.0
        frac_rhythmic = s["rhythmic_count"] / t if t else 0.0
        frac_too_short = s["too_short_count"] / t if t else 0.0

        print(
            f"{source}/{category}/{purity}: {c}/{t} correct ({bucket_acc:.3f}) | "
            f"low={avg_low:.3f}, high={avg_high:.3f}, "
            f"flat={avg_flat:.3f}, flat_hi_frac={avg_high_flat:.3f}, "
            f"peaks={avg_peaks:.2f}, rhythmic%={frac_rhythmic:.2f}, "
            f"too_short%={frac_too_short:.2f}"
        )

if __name__ == "__main__":
    main()
