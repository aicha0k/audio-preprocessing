from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import wave
from sklearn.model_selection import train_test_split

INPUT_ROOT = Path("/home/aicha/Documents/UFSC/26.1/IAnaBorda/archive/mafaulda")
OUTPUT_ROOT = Path("ei_engine_audio_dataset")

SR_ORIGINAL = 50000
SR_TARGET = 16000
MIC_COLUMN_INDEX = 7

WINDOW_SECONDS = 3
OVERLAP = 0.5

MODE = "multiclass"  # "binary" or "multiclass"

np.random.seed(42)


def load_mic_csv(csv_path):
    df = pd.read_csv(csv_path, header=None)
    audio = df.iloc[:, MIC_COLUMN_INDEX].to_numpy(dtype=np.float32)

    audio = audio - np.mean(audio)
    audio = audio / (np.max(np.abs(audio)) + 1e-8)

    return audio


def split_windows(signal, sr, window_seconds=1.0, overlap=0.5):
    window_size = int(sr * window_seconds)
    step = int(window_size * (1 - overlap))

    windows = []
    for start in range(0, len(signal) - window_size + 1, step):
        windows.append(signal[start:start + window_size])

    return windows


def save_wav_int16(path, audio, sr):
    path.parent.mkdir(parents=True, exist_ok=True)

    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16 bits
        wf.setframerate(sr)
        wf.writeframes(audio_int16.tobytes())


csv_files = sorted(INPUT_ROOT.rglob("*.csv"))

# file-level split, to reduce leakage
labels_per_file = []
for csv_path in csv_files:
    original_label = csv_path.relative_to(INPUT_ROOT).parts[0]

    if MODE == "binary":
        label = "normal" if original_label == "normal" else "faulty"
    else:
        label = original_label

    labels_per_file.append(label)

train_files, test_files = train_test_split(
    csv_files,
    test_size=0.2,
    random_state=42,
    stratify=labels_per_file
)

file_to_category = {str(p): "training" for p in train_files}
file_to_category.update({str(p): "testing" for p in test_files})

counter = {}

for csv_path in csv_files:
    original_label = csv_path.relative_to(INPUT_ROOT).parts[0]

    if MODE == "binary":
        label = "normal" if original_label == "normal" else "faulty"
    else:
        label = original_label

    category = file_to_category[str(csv_path)]

    audio = load_mic_csv(csv_path)

    # resample to the same rate used by the XIAO microphone project
    audio_16k = librosa.resample(
        audio,
        orig_sr=SR_ORIGINAL,
        target_sr=SR_TARGET
    )

    windows = split_windows(
        audio_16k,
        sr=SR_TARGET,
        window_seconds=WINDOW_SECONDS,
        overlap=OVERLAP
    )

    for window in windows:
        key = (category, label)
        counter[key] = counter.get(key, 0) + 1

        out_name = f"{label}_{counter[key]:05d}.wav"
        out_path = OUTPUT_ROOT / category / label / out_name

        save_wav_int16(out_path, window, SR_TARGET)

print("Done.")
for key, value in sorted(counter.items()):
    print(key, value)