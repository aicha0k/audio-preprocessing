# Audio Preprocessing and Fault Classification

This repository contains an audio preprocessing and machine learning pipeline for fault classification in rotating machinery using the MAFAULDA dataset.

The project was developed as part of a TinyML study focused on acoustic fault diagnosis and future deployment on the XIAO ESP32S3 Sense.

## Pipeline

The implemented pipeline performs the following steps:

1. Extracts the microphone channel from the original MAFAULDA CSV files;
2. Normalizes the audio signal;
3. Resamples the signal from 50 kHz to 16 kHz;
4. Splits the signal into overlapping audio windows;
5. Generates mono 16-bit WAV files;
6. Extracts Log-Mel features;
7. Trains a lightweight convolutional neural network;
8. Converts and quantizes the model to TensorFlow Lite int8.

The model classifies six operating conditions:

- `horizontal-misalignment`
- `imbalance`
- `normal`
- `overhang`
- `underhang`
- `vertical-misalignment`

## Files

- `audio-conv.py`: converts the microphone data from MAFAULDA CSV files into WAV files organized into training and testing sets.
- `logmel_pipeline.ipynb`: contains data exploration, Log-Mel feature extraction, model training, evaluation and TensorFlow Lite conversion.

## Results

The quantized TensorFlow Lite int8 model achieved an accuracy of **78.72%**.

The TensorFlow Lite model was generated successfully, although it has not yet been tested directly on the ESP32-S3.

### Edge Impulse Version


An alternative version of the project was developed using [Edge Impulse](https://studio.edgeimpulse.com/public/1047481/live) and deployed on the XIAO ESP32S3 Sense.