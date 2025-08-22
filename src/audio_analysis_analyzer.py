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
        # If a model path is provided and exists, we could load it here.
        self._has_model = bool(model_path and os.path.exists(model_path))

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
        rms = self._rms(samples)
        zcr = self._zcr(samples)
        centroid = self._spectral_centroid(samples, sr)

        events: List[Dict[str, Any]] = []

        # Rule 1: Loud transient / event
        # Use a dynamic threshold relative to RMS of the segment
        loud_thresh = max(0.02, rms * 1.8)
        if rms >= loud_thresh:
            conf = min(1.0, rms / (loud_thresh + 1e-9))
            events.append({
                "type": "loud_sound",
                "timestamp": float(timestamp),
                "confidence": float(conf),
                "rms": float(rms),
            })

        # Rule 2: speech-like (moderate rms, low centroid, moderate zcr)
        # heuristic thresholds chosen to be permissive; tune for your data
        if (rms > 0.005 and rms < 0.5) and (zcr > 0.001 and zcr < 0.2) and (centroid > 50 and centroid < 4000):
            # confidence increases when centroid is lower and zcr is in speech-like range
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

