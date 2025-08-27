Train audio model

Use the training script to create a simple classifier from labeled WAV files.

Layout:

 dataset/
   loud_sound/
     loud1.wav
   speech_like/
     s1.wav
   high_freq_noise/
     n1.wav

Run (from repo root, with the virtualenv activated):

```
python tools/train_audio_model.py dataset/ --out models/audio_model.joblib
```

This creates `models/audio_model.joblib` and `models/audio_model.joblib.labels.json` which can be passed to the runtime `AudioAnalysisAnalyzer` as the `model_path` parameter.
