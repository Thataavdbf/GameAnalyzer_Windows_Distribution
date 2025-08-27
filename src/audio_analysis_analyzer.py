"""Lightweight rule-based audio analysis for game audio.

This analyzer is intentionally simple and dependency-light. It accepts:
- a numpy array of samples (mono or stereo) or
- a tuple (samples, sample_rate) or
- a pydub.AudioSegment-like object (has get_array_of_samples and frame_rate) or
- a file path to a WAV file (uses scipy.io.wavfile when available)

It returns a list of event dicts like:
  {"type": "loud_sound", "timestamp": 12.3, "confidence": 0.8}

This is a rules-based fallback implementation. For production-quality detection
replace this with a trained model or a library like pyannote, VAD, or an ML model.
"""

from typing import List, Dict, Any
import numpy as np
import os

try:
    # scipy is available in the distribution; use it when loading files
    from scipy.io import wavfile
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


class AudioAnalysisAnalyzer:
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        # If a model path is provided and exists, try to load a model (joblib/sklearn, Keras, or PyTorch)
        self._has_model = False
        self._model = None
        self._label_map = None
        if model_path and os.path.exists(model_path):
            # try joblib (scikit-learn)
            try:
                import joblib
                self._model = joblib.load(model_path)
                self._has_model = True
            except Exception:
                # try Keras (use dynamic import to avoid static analyzers complaining if tensorflow is not installed)
                try:
                    import importlib
                    # prefer tensorflow.keras if available, otherwise try standalone keras
                    try:
                        tf = importlib.import_module('tensorflow')
                        _load_keras = getattr(tf.keras.models, 'load_model')
                    except Exception:
                        keras = importlib.import_module('keras')
                        _load_keras = getattr(keras.models, 'load_model')
                    self._model = _load_keras(model_path)
                    self._has_model = True
                except Exception:
                    # try PyTorch
                    try:
                        import torch
                        self._model = torch.load(model_path)
                        self._has_model = True
                    except Exception:
                        self._has_model = False
            # try to load label map from model_path + '.labels.json'
            try:
                import json
                labels_path = model_path + '.labels.json'
                if os.path.exists(labels_path):
                    with open(labels_path, 'r', encoding='utf-8') as f:
                        self._label_map = json.load(f)
            except Exception:
                self._label_map = None

    def needs_training(self) -> bool:
        """Return True if no model is present (we're using the rule-based fallback)."""
        return not self._has_model

    def _ensure_mono(self, samples: np.ndarray) -> np.ndarray:
        # If stereo or multi-channel, average to mono
        if samples.ndim == 1:
            return samples
        try:
            return np.mean(samples, axis=1)
        except Exception:
            return samples.flatten()

    def _rms(self, samples: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(samples))))

    def _zcr(self, samples: np.ndarray) -> float:
        # Zero-crossing rate (fraction of sign-changes)
        s = samples
        if s.size < 2:
            return 0.0
        signs = np.sign(s)
        # treat zeros as previous sign to avoid spurious crossings
        signs[signs == 0] = 1
        crossings = np.sum(np.abs(np.diff(signs)) > 0)
        return float(crossings) / float(s.size)

    def _spectral_centroid(self, samples: np.ndarray, sr: int) -> float:
        # approximate spectral centroid using rfft
        try:
            mags = np.abs(np.fft.rfft(samples.astype(float)))
            if mags.sum() <= 0:
                return 0.0
            freqs = np.fft.rfftfreq(len(samples), 1.0 / float(sr))
            centroid = float((mags * freqs).sum() / mags.sum())
            return centroid
        except Exception:
            return 0.0

    def _feature_vector(self, samples: np.ndarray, sr: int) -> np.ndarray:
        """Return a numeric feature vector for a given window of samples."""
        rms = self._rms(samples)
        zcr = self._zcr(samples)
        centroid = self._spectral_centroid(samples, sr)
        peak = float(np.max(np.abs(samples))) if samples.size > 0 else 0.0
        # spectral flatness (geometric mean / arithmetic mean)
        try:
            S = np.abs(np.fft.rfft(samples.astype(float)))
            gm = float(np.exp(np.mean(np.log(S + 1e-12))))
            am = float(np.mean(S) + 1e-12)
            flatness = gm / am
        except Exception:
            flatness = 0.0
        return np.array([rms, zcr, centroid, peak, flatness], dtype=float)

    def _load_from_path(self, path: str):
        if SCIPY_AVAILABLE:
            try:
                sr, data = wavfile.read(path)
                # convert ints to float32 in -1..1
                if data.dtype.kind in ("i", "u"):
                    maxv = float(np.iinfo(data.dtype).max)
                    data = data.astype("float32") / maxv
                return data, int(sr)
            except Exception:
                pass
        # Fallback: try wave + numpy (limited to PCM wav)
        try:
            import wave
            with wave.open(path, "rb") as wf:
                sr = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
                import numpy as _np
                dtype = _np.int16
                data = _np.frombuffer(frames, dtype=dtype)
                return data, int(sr)
        except Exception:
            raise RuntimeError(f"Unable to read audio file: {path}")

    def analyze_audio(self, audio_segment: Any, timestamp: float = 0.0) -> List[Dict[str, Any]]:
        """Analyze a short audio segment and return detected audio events.

        audio_segment may be:
        - numpy array (samples)
        - (samples, sr) tuple
        - pydub.AudioSegment-like object (has get_array_of_samples and frame_rate)
        - file path to a wav file

        Returns a list of event dicts.
        """
        samples = None
        sr = None

        # Unpack common formats
        if isinstance(audio_segment, (list, tuple)) and len(audio_segment) >= 2:
            samples, sr = audio_segment[0], int(audio_segment[1])
        elif isinstance(audio_segment, np.ndarray):
            samples = audio_segment
        elif isinstance(audio_segment, str) and os.path.exists(audio_segment):
            samples, sr = self._load_from_path(audio_segment)
        else:
            # pydub.AudioSegment compatibility
            if hasattr(audio_segment, "get_array_of_samples") and hasattr(audio_segment, "frame_rate"):
                try:
                    arr = np.array(audio_segment.get_array_of_samples())
                    samples = arr
                    sr = int(audio_segment.frame_rate)
                except Exception:
                    pass

        if samples is None:
            # Nothing we can analyze
            return []

        # ensure numpy and mono
        samples = np.asarray(samples)
        samples = self._ensure_mono(samples)
        # normalize if integers
        if samples.dtype.kind in ("i", "u"):
            try:
                samples = samples.astype("float32") / float(np.iinfo(samples.dtype).max)
            except Exception:
                samples = samples.astype("float32")

        # default sample rate
        sr = int(sr) if sr else 22050

        # compute simple features
        events: List[Dict[str, Any]] = []

        # If a trained model is available, compute features and run inference.
        if self._has_model and self._model is not None:
            try:
                fv = self._feature_vector(samples, sr).reshape(1, -1)
                # scikit-learn style
                if hasattr(self._model, "predict_proba"):
                    probs = self._model.predict_proba(fv)[0]
                    class_idx = int(probs.argmax())
                    conf = float(probs[class_idx])
                else:
                    # keras or other: predict -> probabilities or logits
                    pred = self._model.predict(fv)
                    # flatten
                    arr = np.asarray(pred).ravel()
                    class_idx = int(arr.argmax())
                    conf = float(arr[class_idx])

                # Resolve label
                label = None
                if self._label_map:
                    label = self._label_map.get(str(class_idx)) or self._label_map.get(class_idx)
                # try model attribute
                if label is None and hasattr(self._model, "classes_"):
                    try:
                        label = str(self._model.classes_[class_idx])
                    except Exception:
                        label = None

                if label is None:
                    # fallback mapping for common labels
                    fallback = {0: "loud_sound", 1: "speech_like", 2: "high_freq_noise"}
                    label = fallback.get(class_idx, f"class_{class_idx}")

                events.append({
                    "type": label,
                    "timestamp": float(timestamp),
                    "confidence": float(conf),
                    "features": fv.flatten().tolist(),
                })
                return events
            except Exception:
                # if model inference fails, fall back to heuristics
                self._has_model = False

        # Heuristic fallback (same as before)
        rms = self._rms(samples)
        zcr = self._zcr(samples)
        centroid = self._spectral_centroid(samples, sr)
        peak = float(np.max(np.abs(samples))) if samples.size > 0 else 0.0

        # Rule 1: Loud transient / event
        if peak > 0.25:
            conf = min(1.0, peak)
            events.append({
                "type": "loud_sound",
                "timestamp": float(timestamp),
                "confidence": float(conf),
                "rms": float(rms),
                "peak": peak,
            })
        else:
            if rms >= 0.02:
                conf = min(1.0, rms / 0.1)
                events.append({
                    "type": "loud_sound",
                    "timestamp": float(timestamp),
                    "confidence": float(conf),
                    "rms": float(rms),
                })

        # Rule 2: speech-like
        if (rms > 0.005 and rms < 0.5) and (zcr > 0.001 and zcr < 0.2) and (centroid > 50 and centroid < 4000):
            conf = 1.0 - (abs(centroid - 1000.0) / 4000.0)
            conf = max(0.0, min(1.0, conf))
            events.append({
                "type": "speech_like",
                "timestamp": float(timestamp),
                "confidence": float(conf),
                "rms": float(rms),
                "zcr": float(zcr),
                "centroid": float(centroid),
            })

        # Rule 3: high-frequency noise
        if centroid > 6000 and rms > 0.01:
            events.append({
                "type": "high_freq_noise",
                "timestamp": float(timestamp),
                "confidence": min(1.0, (centroid - 6000) / 20000.0),
                "centroid": float(centroid),
            })

        return events

