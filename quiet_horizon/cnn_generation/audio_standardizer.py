# quiet_horizon/cnn/audio_standardizer.py

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

# Defaults – change if your CNN uses different settings
TARGET_SR = 22050
WINDOW_SECONDS = 2.5  # length of one CNN window in seconds


def load_and_standardize(
    path: str | Path,
    target_sr: int = TARGET_SR,
    window_seconds: float = WINDOW_SECONDS,
) -> tuple[np.ndarray, int]:
    """
    Load arbitrary audio (wav, mp3, flac, ogg, etc.) and convert to:
      - mono
      - target_sr
      - fixed length window_seconds (center crop or zero-pad)

    Returns:
      y: np.ndarray, shape (samples,)
      sr: int
    """
    path = Path(path)
    y, sr = librosa.load(path.as_posix(), sr=target_sr, mono=True)

    target_len = int(window_seconds * target_sr)

    if len(y) > target_len:
        # center crop
        start = (len(y) - target_len) // 2
        y = y[start : start + target_len]
    elif len(y) < target_len:
        # zero pad at the end
        pad = target_len - len(y)
        y = np.pad(y, (0, pad))

    return y.astype(np.float32), target_sr


def save_wav_for_cnn(
    y: np.ndarray,
    sr: int,
    out_path: str | Path,
) -> None:
    """
    Save standardized audio as 16-bit PCM WAV for the CNN.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path.as_posix(), y, sr, subtype="PCM_16")


def convert_to_cnn_wav(
    in_path: str | Path,
    out_path: str | Path,
    target_sr: int = TARGET_SR,
    window_seconds: float = WINDOW_SECONDS,
) -> None:
    """
    Convenience one-shot: load arbitrary audio, standardize,
    and write CNN-ready WAV.
    """
    y, sr = load_and_standardize(in_path, target_sr, window_seconds)
    save_wav_for_cnn(y, sr, out_path)
