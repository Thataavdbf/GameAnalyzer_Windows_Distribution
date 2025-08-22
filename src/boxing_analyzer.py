import cv2
import numpy as np
from event_logger import EventLogger

class BoxingAnalyzer:
    def __init__(self):
        self.logger = EventLogger()
        self.punch_count = 0
        self.combo_count = 0
        self.head_movement_score = 0
        self.punch_accuracy = 0
        self.total_punches_thrown = 0
        self.total_punches_landed = 0
        self.last_punch_time = 0
        self.combo_window = 2.0  # seconds
        self.current_combo = []
        
    def analyze_frame(self, frame, timestamp):
        """
        Analyze a single frame for Undisputed Boxing specific events
        """
        events = []
        
        # Detect punches
        punch_detected = self.detect_punch(frame, timestamp)
        if punch_detected:
            events.append(punch_detected)
            
        # Detect head movement
        head_movement = self.detect_head_movement(frame, timestamp)
        if head_movement:
            events.append(head_movement)
            
        # Detect combos
        combo_detected = self.detect_combo(timestamp)
        if combo_detected:
            events.append(combo_detected)
            
        # Analyze defense
        defense_analysis = self.analyze_defense(frame, timestamp)
        if defense_analysis:
            events.append(defense_analysis)
            
        # Analyze stamina/fatigue indicators
        stamina_analysis = self.analyze_stamina(frame, timestamp)
        if stamina_analysis:
            events.append(stamina_analysis)
            
        return events
        
    def detect_punch(self, frame, timestamp):
        """
        Detect punch throws and impacts
        """
        # Convert to grayscale for motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Use optical flow or frame differencing to detect rapid movements
        # This is simplified - would need more sophisticated motion analysis
        
        # Look for rapid changes in specific regions (where hands would be)
        height, width = gray.shape
        
        # Define regions of interest for left and right hands
        left_hand_roi = gray[height//3:2*height//3, :width//3]
        right_hand_roi = gray[height//3:2*height//3, 2*width//3:]
        
        # Calculate motion intensity (simplified)
        left_motion = np.std(left_hand_roi)
        right_motion = np.std(right_hand_roi)
        
        punch_threshold = 30  # This would need tuning with real data
        
        punch_type = None
        punch_landed = False
        
        if left_motion > punch_threshold:
            punch_type = "left_hand"
            self.total_punches_thrown += 1
            
            # Check if punch landed (look for impact indicators)
            punch_landed = self.check_punch_impact(frame, "left")
            
        elif right_motion > punch_threshold:
            punch_type = "right_hand"
            self.total_punches_thrown += 1
            
            # Check if punch landed
            punch_landed = self.check_punch_impact(frame, "right")
            
        if punch_type:
            if punch_landed:
                self.total_punches_landed += 1
                
            # Update combo tracking
            self.current_combo.append({
                'type': punch_type,
                'timestamp': timestamp,
                'landed': punch_landed
            })
            
            self.last_punch_time = timestamp
            self.punch_count += 1
            
            # Calculate accuracy
            if self.total_punches_thrown > 0:
                self.punch_accuracy = self.total_punches_landed / self.total_punches_thrown
                
            return {
                'type': 'punch',
                'timestamp': timestamp,
                'description': f'{punch_type} punch {"landed" if punch_landed else "missed"}',
                'punch_type': punch_type,
                'landed': punch_landed,
                'accuracy': self.punch_accuracy
            }
            
        return None
        
    def check_punch_impact(self, frame, hand_side):
        """
        Check if a punch made contact by looking for visual impact indicators
        """
        # Look for impact effects, opponent reaction, etc.
        # This is simplified - would need actual game-specific indicators
        
        # Convert to HSV to look for impact effects
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Look for white/bright flashes that might indicate impact
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, white_lower, white_upper)
        
        white_pixels = cv2.countNonZero(white_mask)
        
        # If there are bright pixels in the impact area, assume punch landed
        return white_pixels > 100
        
    def detect_head_movement(self, frame, timestamp):
        """
        Detect and score head movement for defense
        """
        # Use face detection or head tracking
        # This is simplified - would use more sophisticated tracking
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Use Haar cascades for face detection (basic approach)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Track head position changes
            # For now, just detect if head is moving based on position
            x, y, w, h = faces[0]
            head_center = (x + w//2, y + h//2)
            
            # Calculate movement score based on position relative to center
            frame_center = (frame.shape[1]//2, frame.shape[0]//2)
            distance_from_center = np.sqrt((head_center[0] - frame_center[0])**2 + 
                                         (head_center[1] - frame_center[1])**2)
            
            # Score head movement (more movement from center = better defense)
            movement_score = min(distance_from_center / 100, 1.0)
            self.head_movement_score = (self.head_movement_score + movement_score) / 2
            
            return {
                'type': 'head_movement',
                'timestamp': timestamp,
                'description': f'Head movement detected (score: {movement_score:.2f})',
                'movement_score': movement_score,
                'head_position': head_center
            }
            
        return None
        
    def detect_combo(self, timestamp):
        """
        Detect punch combinations
        """
        # Clean up old punches outside combo window
        self.current_combo = [p for p in self.current_combo 
                            if timestamp - p['timestamp'] <= self.combo_window]
        
        # Check if we have a valid combo (2+ punches within window)
        if len(self.current_combo) >= 2:
            # Check if this is a new combo (not already counted)
            if timestamp - self.last_punch_time <= 0.5:  # Recent punch
                combo_types = [p['type'] for p in self.current_combo]
                combo_landed = sum(1 for p in self.current_combo if p['landed'])
                
                self.combo_count += 1
                
                return {
                    'type': 'combo',
                    'timestamp': timestamp,
                    'description': f'{len(self.current_combo)}-punch combo ({combo_landed} landed)',
                    'combo_length': len(self.current_combo),
                    'combo_types': combo_types,
                    'punches_landed': combo_landed
                }
                
        return None
        
    def analyze_defense(self, frame, timestamp):
        """
        Analyze defensive techniques (blocking, parrying, dodging)
        """
        # Look for defensive postures and movements
        # This is simplified - would need more sophisticated analysis
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Look for high guard position (hands up near face)
        height, width = gray.shape
        upper_region = gray[:height//2, :]
        
        # Use edge detection to find hand/arm positions
        edges = cv2.Canny(upper_region, 50, 150)
        edge_density = np.sum(edges) / (height * width // 2)
        
        # High edge density in upper region might indicate guard position
        if edge_density > 0.1:  # Threshold would need tuning
            return {
                'type': 'defense',
                'timestamp': timestamp,
                'description': 'Defensive posture detected',
                'defense_type': 'high_guard',
                'effectiveness': min(edge_density * 10, 1.0)
            }
            
        return None
        
    def analyze_stamina(self, frame, timestamp):
        """
        Analyze stamina/fatigue indicators from the game UI
        """
        # Look for stamina bars or fatigue indicators in the UI
        # This would need to be customized for the specific game's UI
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Look for green stamina bars (common in games)
        green_lower = np.array([40, 50, 50])
        green_upper = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        
        # Look for red fatigue indicators
        red_lower = np.array([0, 50, 50])
        red_upper = np.array([10, 255, 255])
        red_mask = cv2.inRange(hsv, red_lower, red_upper)
        
        green_pixels = cv2.countNonZero(green_mask)
        red_pixels = cv2.countNonZero(red_mask)
        
        if green_pixels > 500:
            stamina_level = "high"
        elif red_pixels > 500:
            stamina_level = "low"
        else:
            stamina_level = "medium"
            
        return {
            'type': 'stamina',
            'timestamp': timestamp,
            'description': f'Stamina level: {stamina_level}',
            'stamina_level': stamina_level,
            'green_pixels': green_pixels,
            'red_pixels': red_pixels
        }
        
    def get_fight_analysis(self):
        """
        Get comprehensive fight analysis
        """
        return {
            'total_punches_thrown': self.total_punches_thrown,
            'total_punches_landed': self.total_punches_landed,
            'punch_accuracy': self.punch_accuracy,
            'combo_count': self.combo_count,
            'head_movement_score': self.head_movement_score,
            'average_combo_length': len(self.current_combo) if self.current_combo else 0
        }
        
    def get_tactical_suggestions(self):
        """
        Generate tactical suggestions based on analysis
        """
        suggestions = []
        
        if self.punch_accuracy < 0.4:
            suggestions.append("Focus on accuracy over volume - you're missing too many punches")
            
        if self.combo_count < 3:
            suggestions.append("Work on combination punching - single shots are easier to defend")
            
        if self.head_movement_score < 0.3:
            suggestions.append("Improve head movement - you're too stationary and predictable")
            
        if self.total_punches_thrown > 50 and self.combo_count == 0:
            suggestions.append("Learn to chain punches together for more effective combinations")
            
        return suggestions

if __name__ == '__main__':
    # Test the analyzer
    analyzer = BoxingAnalyzer()
    
    # Create a test frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some motion simulation
    cv2.rectangle(test_frame, (100, 150), (150, 200), (255, 255, 255), -1)  # Bright region for impact
    
    events = analyzer.analyze_frame(test_frame, 5.0)
    print("Detected events:", events)
    print("Fight analysis:", analyzer.get_fight_analysis())
    print("Tactical suggestions:", analyzer.get_tactical_suggestions())

