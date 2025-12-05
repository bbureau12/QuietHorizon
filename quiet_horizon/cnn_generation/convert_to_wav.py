# quiet_horizon/cnn/convert_to_wav.py

import argparse
from pathlib import Path

from .audio_standardizer import convert_to_cnn_wav


def main():
    parser = argparse.ArgumentParser(
        description="Convert audio files to QuietHorizon CNN-ready WAV format."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input audio file(s) (wav, mp3, flac, ogg, etc.)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="converted_wav",
        help="Directory to write converted WAV files (default: converted_wav)",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_qh",
        help="Suffix to add before extension (default: _qh)",
    )

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for in_path_str in args.inputs:
        in_path = Path(in_path_str)
        if not in_path.is_file():
            print(f"[WARN] Skipping non-file: {in_path}")
            continue

        stem = in_path.stem
        out_path = out_dir / f"{stem}{args.suffix}.wav"

        print(f"Converting {in_path} -> {out_path}")
        convert_to_cnn_wav(in_path, out_path)

    print("Done.")


if __name__ == "__main__":
    main()
