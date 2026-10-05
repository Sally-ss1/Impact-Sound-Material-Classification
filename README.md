# Impact Sound-Based Material Classification

This repository contains the preprocessing and model-training code used for impact sound-based material classification using spectrogram images and a compact Inception-based convolutional neural network.

The study investigates acoustic sensing as a complementary modality for material identification in low-visibility firefighting and robotic applications.

## Material Categories

The classification task includes ten material categories:

- Plastic
- Metal
- Wood
- Glass
- Concrete
- Carpet
- Foam
- Paper
- Textile
- Soil

More than 3,000 impact sound events were collected from multiple physical objects.

## Repository Structure

This repository contains two main Python scripts:

### `preprocessing.py`

This script performs audio preprocessing and spectrogram generation. It:

- loads the raw `.m4a` impact-sound recordings
- resamples the audio to 44.1 kHz
- detects impact events using RMS energy
- extracts each impact segment from 0.05 s before to 0.20 s after the detected event
- converts each segment into a log-frequency spectrogram
- saves the resulting spectrograms as PNG images

### `train.py`

This script implements the compact Inception-based CNN used for material classification. It:

- loads the spectrogram images
- resizes them to 224 × 224 pixels
- normalizes pixel values to [0, 1]
- splits the intra-object data into training and validation subsets
- trains the Inception-based CNN
- applies early stopping and learning-rate reduction
- evaluates classification performance
- generates training curves, classification metrics, and a confusion matrix

## Usage

Run the preprocessing script first:

```bash
python preprocessing.py
```

Then run the training script:

```bash
python train.py
```

## Software Environment

The code was implemented using:

- Python 3.11
- TensorFlow 2.20.0
- Keras 3.13.2
