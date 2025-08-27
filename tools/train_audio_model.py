"""Simple training pipeline for audio event classifier.

Expect a directory structure like:
  dataset/
    loud_sound/
      file1.wav
      file2.wav
    speech_like/
      file3.wav
    high_freq_noise/
      file4.wav

This script extracts the feature vectors used by the runtime analyzer and trains
a scikit-learn RandomForest classifier, saving the model and labels mapping.
"""
import os
import argparse
import json
import numpy as np
from joblib import dump

from src.audio_analysis_analyzer import AudioAnalysisAnalyzer


def iter_wav_files(root_dir):
    for class_name in sorted(os.listdir(root_dir)):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith('.wav'):
                yield class_name, os.path.join(class_dir, fname)


def load_wav_as_array(path):
    # Try scipy.io.wavfile, fallback to wave
    try:
        from scipy.io import wavfile
        sr, data = wavfile.read(path)
        data = data.astype(float)
        # normalize to -1..1 if integer
        if data.dtype.kind in 'iu':
            maxv = float(max(abs(data.min()), abs(data.max()), 1))
            data = data / maxv
        return sr, data
    except Exception:
        import wave, struct
        with wave.open(path, 'rb') as wf:
            sr = wf.getframerate()
            n = wf.getnframes()
            sampwidth = wf.getsampwidth()
            frames = wf.readframes(n)
            fmt = '<' + ('h' if sampwidth == 2 else 'i') * n
            data = np.array(struct.unpack(fmt, frames), dtype=float)
            maxv = float(max(abs(data.min()), abs(data.max()), 1))
            data = data / maxv
            return sr, data


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dataset_dir', help='Root dataset directory with class subfolders')
    p.add_argument('--out', default='audio_model.joblib', help='Output model filename')
    p.add_argument('--labels', default=None, help='Optional labels JSON output (defaults to <out>.labels.json)')
    args = p.parse_args()

    analyzer = AudioAnalysisAnalyzer()

    X = []
    y = []
    label_to_idx = {}
    for class_name, path in iter_wav_files(args.dataset_dir):
        if class_name not in label_to_idx:
            label_to_idx[class_name] = len(label_to_idx)
        idx = label_to_idx[class_name]
        try:
            sr, samples = load_wav_as_array(path)
            # make sure mono
            if samples.ndim > 1:
                samples = np.mean(samples, axis=1)
            fv = analyzer._feature_vector(np.array(samples), int(sr))
            X.append(fv)
            y.append(idx)
            print(f"Loaded {path} -> class {class_name}")
        except Exception as e:
            print(f"Failed to load {path}: {e}")

    X = np.asarray(X)
    y = np.asarray(y)

    if len(X) == 0:
        print("No training data found; exiting")
        return

    # Train a classifier
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X, y)

    out_path = args.out
    dump(clf, out_path)
    labels_path = args.labels or (out_path + '.labels.json')
    inv_map = {str(v): k for k, v in label_to_idx.items()}
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump(inv_map, f, indent=2)
    print(f"Saved model to {out_path} and labels to {labels_path}")


if __name__ == '__main__':
    main()
