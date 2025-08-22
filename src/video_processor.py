import cv2
# moviepy 2.x may not provide a top-level 'editor' module in some installs.
# Import VideoFileClip from the package root where __init__ exposes it.
from moviepy import VideoFileClip

class VideoProcessor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.clip = VideoFileClip(video_path)

    def get_frame_count(self):
        return int(self.clip.fps * self.clip.duration)

    def get_fps(self):
        return self.clip.fps

    def get_duration(self):
        return self.clip.duration

    def extract_frame(self, frame_time_in_seconds):
        return self.clip.get_frame(frame_time_in_seconds)

    def close(self):
        self.clip.close()

    def process_video_with_opencv(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Process frame here (e.g., display, analyze)
            # For now, just showing a placeholder for processing
            # print(f"Processing frame...")

        cap.release()
        print("Video processing complete.")

if __name__ == '__main__':
    # Example usage (replace with a valid video path)
    # For testing, you might need a dummy video file
    # video_file = "path/to/your/video.mp4"
    # processor = VideoProcessor(video_file)
    # print(f"Frame count: {processor.get_frame_count()}")
    # print(f"FPS: {processor.get_fps()}")
    # print(f"Duration: {processor.get_duration()} seconds")
    # frame = processor.extract_frame(10) # Get frame at 10 seconds
    # if frame is not None:
    #     print(f"Extracted frame with shape: {frame.shape}")
    # processor.process_video_with_opencv()
    # processor.close()
    pass


