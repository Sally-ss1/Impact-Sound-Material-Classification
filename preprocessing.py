"""
Audio preprocessing script for impact sound-based material classification.

This script:
1. Loads .m4a audio recordings.
2. Resamples audio to 44.1 kHz.
3. Detects impact events using RMS energy.
4. Extracts a 0.25 s segment around each detected event.
5. Converts each segment into a log-frequency spectrogram.
6. Saves the spectrogram as a PNG image.

Environment:
    Python 3.11
"""

import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Configuration
# ============================================================

INPUT_DIR = "./sounddata"
OUTPUT_DIR = "./data0310_G1"

SAMPLE_RATE = 44100
N_FFT = 2048
HOP_LENGTH = 128
FRAME_LENGTH = 2048

DB_MIN = -80
DB_MAX = 0


# ============================================================
# Audio preprocessing
# ============================================================

def process_audio_file(
    filepath,
    category,
    knock_count,
    sr=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    frame_length=FRAME_LENGTH,
    vmin=DB_MIN,
    vmax=DB_MAX,
):
    """
    Detect individual impact events and generate spectrogram images.

    Parameters
    ----------
    filepath : str
        Path to the input audio file.
    category : str
        Material category identifier.
    knock_count : int
        Running count of detected impacts for the category.
    sr : int
        Target sampling rate.
    n_fft : int
        FFT size used for STFT.
    hop_length : int
        Hop length used for RMS calculation and STFT.
    frame_length : int
        Frame length used for RMS energy calculation.
    vmin : float
        Minimum display value of the spectrogram in dB.
    vmax : float
        Maximum display value of the spectrogram in dB.

    Returns
    -------
    int
        Updated impact count.
    """

    # Load audio and resample to the target sampling rate.
    y, sr = librosa.load(filepath, sr=sr)

    # Compute RMS energy.
    energy = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]

    # Adaptive threshold for detecting high-energy impact frames.
    threshold = np.mean(energy) + 2 * np.std(energy)

    frames = np.nonzero(
        energy > threshold
    )[0]

    # Convert frame indices to time.
    times = librosa.frames_to_time(
        frames,
        sr=sr,
        hop_length=hop_length,
    )

    # Detect individual impact events.
    segments = []

    if len(times) > 0:
        detected = False

        for i in range(1, len(times)):

            # Start a new event when the time interval exceeds 0.2 s
            # or when no event is currently active.
            if times[i] - times[i - 1] > 0.2 or not detected:

                start = max(
                    times[i] - 0.05,
                    0,
                )

                end = min(
                    times[i] + 0.2,
                    len(y) / sr,
                )

                segments.append(
                    (start, end)
                )

                detected = True

            # Reset the detection state after a gap longer than 1 s.
            if times[i] - times[i - 1] > 1.0:
                detected = False

    # Generate a spectrogram for each detected impact.
    for start, end in segments:

        start_sample = int(
            start * sr
        )

        end_sample = int(
            end * sr
        )

        y_segment = y[
            start_sample:end_sample
        ]

        # Skip empty or zero-valued segments.
        if (
            len(y_segment) == 0
            or np.all(y_segment == 0)
        ):
            continue

        # Short-time Fourier transform.
        stft = librosa.stft(
            y_segment,
            n_fft=n_fft,
            hop_length=hop_length,
        )

        # Convert magnitude to decibel scale
        # relative to the maximum amplitude.
        spectrogram_db = librosa.amplitude_to_db(
            np.abs(stft),
            ref=np.max,
        )

        # Create spectrogram image.
        plt.figure(
            figsize=(224 / 100, 224 / 100),
            dpi=100,
        )

        librosa.display.specshow(
            spectrogram_db,
            sr=sr,
            hop_length=hop_length,
            x_axis="time",
            y_axis="log",
            vmin=vmin,
            vmax=vmax,
        )

        plt.axis("off")

        knock_count += 1

        output_path = os.path.join(
            OUTPUT_DIR,
            f"{category}_{knock_count}.png",
        )

        os.makedirs(
            OUTPUT_DIR,
            exist_ok=True,
        )

        plt.savefig(
            output_path,
            bbox_inches="tight",
            pad_inches=0,
        )

        plt.close()

        print(
            f"Saved spectrogram: {output_path}"
        )

    return knock_count


# ============================================================
# Directory processing
# ============================================================

def process_directory(
    directory,
    sr=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    frame_length=FRAME_LENGTH,
    vmin=DB_MIN,
    vmax=DB_MAX,
):
    """
    Process all .m4a files in a directory.
    """

    category_knock_counts = {}

    for filename in sorted(
        os.listdir(directory)
    ):

        if not filename.lower().endswith(
            ".m4a"
        ):
            continue

        filepath = os.path.join(
            directory,
            filename,
        )

        base_filename = os.path.splitext(
            filename
        )[0]

        category, physical_number = (
            base_filename.split("_")
        )

        if category not in category_knock_counts:
            category_knock_counts[
                category
            ] = 0

        category_knock_counts[
            category
        ] = process_audio_file(
            filepath,
            category,
            category_knock_counts[
                category
            ],
            sr,
            n_fft,
            hop_length,
            frame_length,
            vmin,
            vmax,
        )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    process_directory(
        INPUT_DIR,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        frame_length=FRAME_LENGTH,
        vmin=DB_MIN,
        vmax=DB_MAX,
    )