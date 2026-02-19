"""
Generate simple test audio files for QuietHorizon testing.

This script creates basic WAV files for testing purposes.
Not needed for unit tests, but useful for integration tests.
"""
import numpy as np
from scipy.io import wavfile
from pathlib import Path


def generate_sine_wave(frequency, duration, sr=22050):
    """Generate a sine wave."""
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * frequency * t)
    return audio, sr


def generate_noise(duration, sr=22050):
    """Generate white noise."""
    audio = np.random.randn(int(sr * duration))
    return audio, sr


def generate_chirp(duration, f0=200, f1=800, sr=22050):
    """Generate a frequency sweep (chirp)."""
    t = np.linspace(0, duration, int(sr * duration))
    # Linear chirp
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration))
    audio = np.sin(phase)
    return audio, sr


def save_wav(filename, audio, sr):
    """Save audio as 16-bit WAV file."""
    # Normalize to [-1, 1]
    audio = audio / np.max(np.abs(audio))
    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)
    wavfile.write(filename, sr, audio_int16)
    print(f"Created: {filename}")


def main():
    """Generate all test audio files."""
    fixtures_dir = Path(__file__).parent
    
    print("Generating test audio files...")
    
    # 1. Sample nature sound (bird-like chirp)
    audio, sr = generate_chirp(duration=3.0, f0=2000, f1=3500)
    save_wav(fixtures_dir / "sample_nature.wav", audio, sr)
    
    # 2. Sample anthro sound (low frequency rumble)
    audio, sr = generate_sine_wave(frequency=120, duration=3.0)
    # Add some harmonics to make it more realistic
    audio += 0.5 * np.sin(2 * np.pi * 240 * np.linspace(0, 3, len(audio)))
    save_wav(fixtures_dir / "sample_anthro.wav", audio, sr)
    
    # 3. Short audio
    audio, sr = generate_sine_wave(frequency=440, duration=0.3)
    save_wav(fixtures_dir / "sample_short.wav", audio, sr)
    
    # 4. Silent audio
    audio = np.zeros(int(22050 * 2))
    save_wav(fixtures_dir / "sample_silent.wav", audio, 22050)
    
    # 5. Mixed nature sounds (multiple frequencies)
    audio1, sr = generate_sine_wave(frequency=440, duration=2.0)
    audio2, _ = generate_sine_wave(frequency=880, duration=2.0)
    audio3, _ = generate_chirp(duration=2.0, f0=1000, f1=2000)
    audio = 0.4 * audio1 + 0.3 * audio2 + 0.3 * audio3
    save_wav(fixtures_dir / "sample_mixed.wav", audio, sr)
    
    # 6. Different formats (for testing)
    # Note: Requires pydub for MP3/OGG conversion
    try:
        from pydub import AudioSegment
        
        # Convert one WAV to other formats
        sound = AudioSegment.from_wav(fixtures_dir / "sample_nature.wav")
        sound.export(fixtures_dir / "sample_nature.mp3", format="mp3")
        print(f"Created: sample_nature.mp3")
        
        sound.export(fixtures_dir / "sample_nature.ogg", format="ogg")
        print(f"Created: sample_nature.ogg")
        
    except ImportError:
        print("Note: Install pydub for MP3/OGG generation")
        print("  pip install pydub")
    
    print("\nTest audio generation complete!")
    print(f"Files saved to: {fixtures_dir}")


if __name__ == "__main__":
    main()
