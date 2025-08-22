import unittest
import sys
import os
import numpy as np
import cv2
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from helldivers_analyzer import HellDiversAnalyzer
from boxing_analyzer import BoxingAnalyzer
from statistics_engine import StatisticsEngine

class TestHellDiversAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = HellDiversAnalyzer()
        
    def test_analyzer_initialization(self):
        """Test analyzer initialization"""
        self.assertIsNotNone(self.analyzer)
        self.assertEqual(self.analyzer.friendly_fire_count, 0)
        self.assertEqual(len(self.analyzer.stratagem_usage), 0)
        
    def test_frame_analysis(self):
        """Test frame analysis with dummy data"""
        # Create a test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add some colored regions to simulate game elements
        cv2.rectangle(test_frame, (100, 100), (200, 200), (0, 0, 255), -1)  # Red damage
        cv2.rectangle(test_frame, (300, 300), (400, 400), (255, 0, 0), -1)  # Blue team
        
        events = self.analyzer.analyze_frame(test_frame, 10.5)
        self.assertIsInstance(events, list)
        
    def test_analysis_summary(self):
        """Test analysis summary generation"""
        summary = self.analyzer.get_analysis_summary()
        self.assertIsInstance(summary, dict)
        self.assertIn('friendly_fire_incidents', summary)
        self.assertIn('stratagem_usage', summary)

class TestBoxingAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = BoxingAnalyzer()
        
    def test_analyzer_initialization(self):
        """Test analyzer initialization"""
        self.assertIsNotNone(self.analyzer)
        self.assertEqual(self.analyzer.punch_count, 0)
        self.assertEqual(self.analyzer.combo_count, 0)
        
    def test_frame_analysis(self):
        """Test frame analysis with dummy data"""
        # Create a test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add some motion simulation
        cv2.rectangle(test_frame, (100, 150), (150, 200), (255, 255, 255), -1)  # Bright region for impact
        
        events = self.analyzer.analyze_frame(test_frame, 5.0)
        self.assertIsInstance(events, list)
        
    def test_fight_analysis(self):
        """Test fight analysis generation"""
        analysis = self.analyzer.get_fight_analysis()
        self.assertIsInstance(analysis, dict)
        self.assertIn('total_punches_thrown', analysis)
        self.assertIn('punch_accuracy', analysis)
        
    def test_tactical_suggestions(self):
        """Test tactical suggestions generation"""
        suggestions = self.analyzer.get_tactical_suggestions()
        self.assertIsInstance(suggestions, list)

class TestStatisticsEngine(unittest.TestCase):
    def setUp(self):
        self.engine = StatisticsEngine()
        
    def test_engine_initialization(self):
        """Test statistics engine initialization"""
        self.assertIsNotNone(self.engine)
        self.assertEqual(len(self.engine.events), 0)
        
    def test_load_events(self):
        """Test loading events"""
        sample_events = [
            {'type': 'punch', 'timestamp': 1.0, 'punch_type': 'left_hand', 'landed': True},
            {'type': 'combo', 'timestamp': 3.0, 'combo_length': 3, 'punches_landed': 2}
        ]
        
        self.engine.load_events(sample_events, 'undisputed_boxing')
        self.assertEqual(len(self.engine.events), 2)
        self.assertEqual(self.engine.game_type, 'undisputed_boxing')
        
    def test_statistics_generation(self):
        """Test statistics generation"""
        sample_events = [
            {'type': 'punch', 'timestamp': 1.0, 'punch_type': 'left_hand', 'landed': True},
            {'type': 'combo', 'timestamp': 3.0, 'combo_length': 3, 'punches_landed': 2}
        ]
        
        self.engine.load_events(sample_events, 'undisputed_boxing')
        stats = self.engine.generate_statistical_overlays()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('punch_analysis', stats)
        self.assertIn('combo_effectiveness', stats)

if __name__ == '__main__':
    # Import cv2 here to avoid issues if not available
    try:
        import cv2
        unittest.main()
    except ImportError:
        print("OpenCV not available, skipping tests that require it")
        # Run tests that don't require cv2
        suite = unittest.TestSuite()
        suite.addTest(TestStatisticsEngine('test_engine_initialization'))
        suite.addTest(TestStatisticsEngine('test_load_events'))
        suite.addTest(TestStatisticsEngine('test_statistics_generation'))
        runner = unittest.TextTestRunner()
        runner.run(suite)

