import os
from quiet_horizon_dsp import QuietHorizonDSP  # adjust if your file/module name differs

# Root of the exported DSP dataset
DATASET_DIR = "quiet_horizon/dsp_dataset"  # the folder we populated earlier, e.g. dsp_dataset/road_noise/pure

def iter_audio_files(root: str):
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith((".wav", ".flac", ".mp3", ".ogg")):
                yield os.path.join(dirpath, name)

def main():
    dsp = QuietHorizonDSP()

    total = 0
    correct = 0

    # Optional: per-bucket stats (category / purity)
    bucket_stats = {}  # (category, purity) -> {total, correct}

    for path in iter_audio_files(DATASET_DIR):
        # Expect structure dsp_dataset/<category>/<purity>/filename.wav
        rel = os.path.relpath(path, DATASET_DIR)
        parts = rel.split(os.sep)

        if len(parts) < 3:
            # Not in category/purity/file form; skip
            category = "unknown"
            purity = "unknown"
        else:
            category = parts[0]
            purity = parts[1]

        expected_anthro = True  # all of these are non-animal noise buckets

        result = dsp.analyze(path)
        predicted_anthro = result["anthropogenic"]

        total += 1
        if expected_anthro == predicted_anthro:
            correct += 1

        key = (category, purity)
        if key not in bucket_stats:
            bucket_stats[key] = {"total": 0, "correct": 0}
        bucket_stats[key]["total"] += 1
        if expected_anthro == predicted_anthro:
            bucket_stats[key]["correct"] += 1

        # You can uncomment this if you want to see individual misfires:
        # if not predicted_anthro:
        #     print(f"[MISS] {rel} -> anthropogenic={predicted_anthro}")

    print(f"\n=== Overall DSP Performance on DIRTY clips ===")
    if total > 0:
        acc = correct / total
    else:
        acc = 0.0
    print(f"Total files: {total}")
    print(f"Correct anthropogenic detections: {correct}")
    print(f"Accuracy: {acc:.3f}")

    print("\n=== Per-bucket stats (category/purity) ===")
    for (category, purity), stats in sorted(bucket_stats.items()):
        t = stats["total"]
        c = stats["correct"]
        bucket_acc = c / t if t > 0 else 0.0
        print(f"{category}/{purity}: {c}/{t} correct ({bucket_acc:.3f})")

if __name__ == "__main__":
    main()
