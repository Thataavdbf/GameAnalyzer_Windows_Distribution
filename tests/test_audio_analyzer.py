import os
import tempfile
import numpy as np
from src.audio_analysis_analyzer import AudioAnalysisAnalyzer


def test_loud_event_detection():
    # synthesize 2s audio: quiet sine + loud click at 1s
    sr = 16000
    t = np.linspace(0, 2, int(2*sr), endpoint=False)
    sine = 0.01 * np.sin(2 * np.pi * 220 * t)
    data = sine.copy()
    # add a loud transient around 1s
    click_start = int(1.0 * sr)
    data[click_start:click_start+200] += 0.8 * np.hanning(200)

    analyzer = AudioAnalysisAnalyzer()
    events = analyzer.analyze_audio((data, sr), 0.0)
    # Expect at least one loud_sound event
    assert any(e.get('type') == 'loud_sound' for e in events)
