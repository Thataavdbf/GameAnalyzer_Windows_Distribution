import os
import sys
from video_processor import VideoProcessor
from object_detector import ObjectDetector
from event_logger import EventLogger


def run(video_path, game_type='test'):
    logger = EventLogger()
    detector = ObjectDetector()
    processor = VideoProcessor(video_path)

    fps = processor.get_fps()
    total_frames = processor.get_frame_count()
    duration = processor.get_duration()

    print(f"Running headless analysis on: {video_path}")
    print(f"FPS: {fps}, Duration: {duration}s, Frames: {total_frames}")

    # Sample every second for a lightweight run
    t = 0.0
    while t < duration:
        frame = processor.extract_frame(t)
        processed_frame, detections = detector.detect_objects(frame)
        if detections:
            for d in detections:
                logger.log_event(t, 'detection', f"{d['class_name']} detected", d)
        t += 1.0

    out_file = f"headless_analysis_{game_type}.json"
    logger.save_events_to_file(out_file)
    processor.close()
    print(f"Headless analysis complete. Results: {out_file}")


if __name__ == '__main__':
    test_video = os.path.join(os.path.dirname(__file__), '..', 'assets', 'test_video.mp4')
    test_video = os.path.abspath(test_video)
    if os.path.exists(test_video):
        run(test_video, 'test')
    else:
        print(f"Test video not found at {test_video}. Please provide a valid video path.")
