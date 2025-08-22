import cv2
import numpy as np
from event_logger import EventLogger

class HellDiversAnalyzer:
    def __init__(self):
        self.logger = EventLogger()
        self.friendly_fire_count = 0
        self.stratagem_usage = {}
        self.objective_status = "unknown"
        
    def analyze_frame(self, frame, timestamp):
        """
        Analyze a single frame for Helldivers 2 specific events
        """
        events = []
        
        # Detect friendly fire (simplified - looking for specific color patterns)
        friendly_fire_detected = self.detect_friendly_fire(frame)
        if friendly_fire_detected:
            self.friendly_fire_count += 1
            events.append({
                'type': 'friendly_fire',
                'timestamp': timestamp,
                'description': 'Friendly fire incident detected',
                'severity': 'high'
            })
            
        # Detect stratagem usage (looking for UI elements)
        stratagem_used = self.detect_stratagem_usage(frame)
        if stratagem_used:
            stratagem_type = stratagem_used.get('type', 'unknown')
            if stratagem_type not in self.stratagem_usage:
                self.stratagem_usage[stratagem_type] = 0
            self.stratagem_usage[stratagem_type] += 1
            
            events.append({
                'type': 'stratagem_usage',
                'timestamp': timestamp,
                'description': f'Stratagem used: {stratagem_type}',
                'stratagem_type': stratagem_type
            })
            
        # Detect objective status
        objective_change = self.detect_objective_status(frame)
        if objective_change:
            events.append({
                'type': 'objective_update',
                'timestamp': timestamp,
                'description': f'Objective status: {objective_change}',
                'status': objective_change
            })
            
        # Squad synergy analysis
        squad_analysis = self.analyze_squad_synergy(frame)
        if squad_analysis:
            events.append({
                'type': 'squad_synergy',
                'timestamp': timestamp,
                'description': squad_analysis['description'],
                'score': squad_analysis['score']
            })
            
        return events
        
    def detect_friendly_fire(self, frame):
        """
        Detect friendly fire incidents by looking for specific visual cues
        This is a simplified implementation - in reality would need more sophisticated CV
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Look for red damage indicators (simplified)
        red_lower = np.array([0, 50, 50])
        red_upper = np.array([10, 255, 255])
        red_mask = cv2.inRange(hsv, red_lower, red_upper)
        
        # Look for blue team indicators
        blue_lower = np.array([100, 50, 50])
        blue_upper = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        
        # If both red damage and blue team indicators are present, might be friendly fire
        red_pixels = cv2.countNonZero(red_mask)
        blue_pixels = cv2.countNonZero(blue_mask)
        
        # Threshold for detection (this would need tuning with real data)
        if red_pixels > 1000 and blue_pixels > 500:
            return True
            
        return False
        
    def detect_stratagem_usage(self, frame):
        """
        Detect when stratagems are being used by looking for UI elements
        """
        # Convert to grayscale for template matching
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Look for stratagem UI elements (this would need actual templates)
        # For now, we'll simulate detection based on certain patterns
        
        # Look for circular patterns that might indicate stratagem selection
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 20,
            param1=50, param2=30, minRadius=10, maxRadius=50
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            if len(circles) > 0:
                # Simulate different stratagem types based on position
                x, y = circles[0][:2]
                if x < frame.shape[1] // 3:
                    return {'type': 'orbital_strike'}
                elif x < 2 * frame.shape[1] // 3:
                    return {'type': 'reinforcement'}
                else:
                    return {'type': 'supply_drop'}
                    
        return None
        
    def detect_objective_status(self, frame):
        """
        Detect objective completion or status changes
        """
        # Look for objective markers or completion indicators
        # This is simplified - would need actual game UI analysis
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Look for green completion indicators
        green_lower = np.array([40, 50, 50])
        green_upper = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        
        green_pixels = cv2.countNonZero(green_mask)
        
        if green_pixels > 2000:
            return "completed"
        elif green_pixels > 500:
            return "in_progress"
        else:
            return "pending"
            
    def analyze_squad_synergy(self, frame):
        """
        Analyze squad coordination and synergy
        """
        # Look for multiple players in frame working together
        # This is a simplified implementation
        
        # Use edge detection to find player silhouettes
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours that might represent players
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size (assuming players have certain size range)
        player_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 1000 < area < 10000:  # Adjust based on actual game footage
                player_contours.append(contour)
                
        if len(player_contours) >= 2:
            # Calculate distances between players
            centers = []
            for contour in player_contours:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centers.append((cx, cy))
                    
            if len(centers) >= 2:
                # Calculate average distance between players
                total_distance = 0
                pairs = 0
                for i in range(len(centers)):
                    for j in range(i + 1, len(centers)):
                        dist = np.sqrt((centers[i][0] - centers[j][0])**2 + 
                                     (centers[i][1] - centers[j][1])**2)
                        total_distance += dist
                        pairs += 1
                        
                avg_distance = total_distance / pairs if pairs > 0 else 0
                
                # Score based on distance (closer = better synergy, but not too close)
                if 50 < avg_distance < 200:
                    score = 0.9
                    description = "Excellent squad formation"
                elif 200 < avg_distance < 400:
                    score = 0.7
                    description = "Good squad spacing"
                else:
                    score = 0.4
                    description = "Poor squad coordination"
                    
                return {
                    'score': score,
                    'description': description,
                    'player_count': len(centers)
                }
                
        return None
        
    def get_analysis_summary(self):
        """
        Get a summary of the analysis results
        """
        return {
            'friendly_fire_incidents': self.friendly_fire_count,
            'stratagem_usage': self.stratagem_usage,
            'total_stratagems_used': sum(self.stratagem_usage.values()),
            'most_used_stratagem': max(self.stratagem_usage.items(), key=lambda x: x[1])[0] if self.stratagem_usage else None
        }

if __name__ == '__main__':
    # Test the analyzer with a dummy frame
    analyzer = HellDiversAnalyzer()
    
    # Create a test frame (normally this would come from video)
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some colored regions to simulate game elements
    cv2.rectangle(test_frame, (100, 100), (200, 200), (0, 0, 255), -1)  # Red damage
    cv2.rectangle(test_frame, (300, 300), (400, 400), (255, 0, 0), -1)  # Blue team
    
    events = analyzer.analyze_frame(test_frame, 10.5)
    print("Detected events:", events)
    print("Analysis summary:", analyzer.get_analysis_summary())

