"""Extract overlapping audio windows from a video file and save as WAVs.

Usage example:
    python tools/extract_audio_windows.py "path/to/video.mp4" out_dataset --window 2.0 --stride 1.0 --label unlabeled

This writes WAV files to out_dataset/<label>/ segment_<start_ms>.wav
"""
import os
import argparse
import math
import numpy as np
from scipy.io import wavfile

def ensure_dir(p):
    try:
        os.makedirs(p, exist_ok=True)
    except Exception:
        pass


def write_wav(path, sr, data):
    # data expected as float32 in -1..1
    # convert to int16
    try:
        clipped = np.clip(data, -1.0, 1.0)
        int_data = (clipped * 32767).astype(np.int16)
        wavfile.write(path, int(sr), int_data)
    except Exception as e:
        raise


def main():
    p = argparse.ArgumentParser(description='Extract overlapping audio windows from a video')
    p.add_argument('video_path', help='Path to video file')
    p.add_argument('out_dir', help='Output dataset directory')
    p.add_argument('--window', type=float, default=2.0, help='Window length in seconds')
    p.add_argument('--stride', type=float, default=1.0, help='Stride (hop) in seconds')
    p.add_argument('--sr', type=int, default=16000, help='Target sample rate')
    p.add_argument('--label', default='unlabeled', help='Class label folder to save windows under')
    args = p.parse_args()

    video_path = args.video_path
    out_dir = args.out_dir
    window_s = float(args.window)
    stride_s = float(args.stride)
    sr = int(args.sr)
    label = args.label or 'unlabeled'

    if not os.path.exists(video_path):
        print('video not found:', video_path)
        return

    # Try to extract audio using ffmpeg (preferred) and read WAV; fall back to moviepy if ffmpeg not available.
    import tempfile
    import subprocess
    import shutil

    temp_wav = None
    audio_mono = None

    ffmpeg_exe = shutil.which('ffmpeg')
    if ffmpeg_exe:
        try:
            tmpf = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_wav = tmpf.name
            tmpf.close()
            cmd = [
                ffmpeg_exe, '-y', '-i', video_path,
                '-vn', '-ac', '1', '-ar', str(sr),
                '-acodec', 'pcm_s16le', temp_wav
            ]
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            read_sr, data = wavfile.read(temp_wav)
            # Convert to float32 in -1..1 range
            if data.dtype == np.int16:
                audio_mono = data.astype(np.float32) / 32767.0
            elif data.dtype == np.int32:
                audio_mono = data.astype(np.float32) / 2147483647.0
            else:
                audio_mono = data.astype(np.float32)
        except Exception as e:
            print('ffmpeg extraction failed:', e)
            audio_mono = None

    if audio_mono is None:
        # Fallback to moviepy if ffmpeg not usable
        try:
            import importlib
            try:
                mpy = importlib.import_module('moviepy.editor')
            except ImportError as ie:
                raise ImportError("moviepy is not installed") from ie
            VideoFileClip = getattr(mpy, 'VideoFileClip')
            clip = VideoFileClip(video_path)
            if not hasattr(clip, 'audio') or clip.audio is None:
                print('No audio track found in video')
                return
            print(f'Extracting audio from {video_path} at {sr}Hz (moviepy fallback)')
            audio_arr = clip.audio.to_soundarray(fps=sr)
            import numpy as _np
            if getattr(audio_arr, 'ndim', 1) == 2:
                audio_mono = _np.mean(audio_arr, axis=1)
            else:
                audio_mono = _np.array(audio_arr)
        except Exception as e:
            print('moviepy required or ffmpeg must be installed:', e)
            return
    else:
        print(f'Extracting audio from {video_path} at {sr}Hz (ffmpeg)')

    # clean up temporary file if created
    if temp_wav:
        try:
            os.remove(temp_wav)
        except Exception:
            pass

    total_samples = len(audio_mono)
    win = int(window_s * sr)
    hop = int(stride_s * sr)
    if win <= 0:
        win = int(2.0 * sr)
    if hop <= 0:
        hop = int(1.0 * sr)

    ensure_dir(out_dir)
    label_dir = os.path.join(out_dir, label)
    ensure_dir(label_dir)

    n_windows = max(1, int(math.floor((total_samples - win) / hop)) + 1)
    print(f'Audio length {total_samples/sr:.2f}s -> {n_windows} windows (win={win/sr}s hop={hop/sr}s)')

    for i in range(n_windows):
        start_idx = i * hop
        end_idx = start_idx + win
        if end_idx > total_samples:
            end_idx = total_samples
        seg = audio_mono[start_idx:end_idx]
        start_ms = int((start_idx / sr) * 1000.0)
        fname = f'segment_{start_ms:08d}.wav'
        out_path = os.path.join(label_dir, fname)
        try:
            write_wav(out_path, sr, seg)
        except Exception as e:
            print('Failed to write', out_path, e)
        else:
            print('Wrote', out_path)

    print('done')


if __name__ == '__main__':
    main()
