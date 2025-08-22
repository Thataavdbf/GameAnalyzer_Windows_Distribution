import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from video_processor import VideoProcessor
import cv2

class TestVideoProcessor(unittest.TestCase):
    def setUp(self):
        self.test_video_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'test_video.mp4')
        
    def test_video_loading(self):
        """Test if video can be loaded successfully"""
        if os.path.exists(self.test_video_path):
            processor = VideoProcessor(self.test_video_path)
            self.assertIsNotNone(processor.clip)
            self.assertGreater(processor.get_duration(), 0)
            processor.close()
        else:
            self.skipTest("Test video not found")
            
    def test_frame_extraction(self):
        """Test frame extraction functionality"""
        if os.path.exists(self.test_video_path):
            processor = VideoProcessor(self.test_video_path)
            frame = processor.extract_frame(1.0)  # Extract frame at 1 second
            self.assertIsNotNone(frame)
            self.assertEqual(len(frame.shape), 3)  # Should be a color image
            processor.close()
        else:
            self.skipTest("Test video not found")
            
    def test_video_properties(self):
        """Test video property retrieval"""
        if os.path.exists(self.test_video_path):
            processor = VideoProcessor(self.test_video_path)
            fps = processor.get_fps()
            duration = processor.get_duration()
            frame_count = processor.get_frame_count()
            
            self.assertGreater(fps, 0)
            self.assertGreater(duration, 0)
            self.assertGreater(frame_count, 0)
            
            processor.close()
        else:
            self.skipTest("Test video not found")

if __name__ == '__main__':
    unittest.main()

