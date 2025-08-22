import cv2
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips
import json
from datetime import datetime

class HighlightGenerator:
    def __init__(self, video_path):
        self.video_path = video_path
        self.clip = VideoFileClip(video_path)
        self.highlights = []
        
    def generate_highlights(self, events, game_type, highlight_types=None):
        """
        Generate highlight clips based on analyzed events
        """
        if highlight_types is None:
            highlight_types = self.get_default_highlight_types(game_type)
            
        self.highlights = []
        
        for event in events:
            if self.should_include_in_highlights(event, highlight_types):
                highlight = self.create_highlight_clip(event)
                if highlight:
                    self.highlights.append(highlight)
                    
        return self.highlights
        
    def get_default_highlight_types(self, game_type):
        """Get default highlight types for each game"""
        if game_type.lower() == 'helldivers_2':
            return [
                'friendly_fire',
                'stratagem_usage',
                'objective_update',
                'squad_synergy'
            ]
        elif game_type.lower() == 'undisputed_boxing':
            return [
                'combo',
                'punch',
                'defense'
            ]
        else:
            return ['all']
            
    def should_include_in_highlights(self, event, highlight_types):
        """Determine if an event should be included in highlights"""
        event_type = event.get('type', '')
        
        # Include all events if 'all' is specified
        if 'all' in highlight_types:
            return True
            
        # Check if event type is in highlight types
        if event_type in highlight_types:
            return True
            
        # Special conditions for different event types
        if event_type == 'punch' and event.get('landed', False):
            return 'punch' in highlight_types
            
        if event_type == 'combo' and event.get('combo_length', 0) >= 3:
            return 'combo' in highlight_types
            
        if event_type == 'friendly_fire':
            return 'friendly_fire' in highlight_types
            
        if event_type == 'stratagem_usage':
            return 'stratagem_usage' in highlight_types
            
        return False
        
    def create_highlight_clip(self, event):
        """Create a highlight clip for a specific event"""
        timestamp = event.get('timestamp', 0)
        event_type = event.get('type', 'unknown')
        
        # Define clip duration based on event type
        clip_durations = {
            'punch': 3.0,
            'combo': 5.0,
            'friendly_fire': 4.0,
            'stratagem_usage': 6.0,
            'objective_update': 5.0,
            'squad_synergy': 4.0,
            'defense': 3.0
        }
        
        duration = clip_durations.get(event_type, 3.0)
        
        # Calculate start and end times
        start_time = max(0, timestamp - duration/2)
        end_time = min(self.clip.duration, timestamp + duration/2)
        
        # Ensure minimum clip length
        if end_time - start_time < 2.0:
            end_time = min(self.clip.duration, start_time + 2.0)
            
        try:
            # Extract the clip
            highlight_clip = self.clip.subclip(start_time, end_time)
            
            # Add text overlay with event description
            description = event.get('description', f'{event_type} at {timestamp:.1f}s')
            
            highlight_info = {
                'clip': highlight_clip,
                'start_time': start_time,
                'end_time': end_time,
                'event_type': event_type,
                'description': description,
                'timestamp': timestamp,
                'metadata': event
            }
            
            return highlight_info
            
        except Exception as e:
            print(f"Error creating highlight clip: {e}")
            return None
            
    def create_highlight_reel(self, output_path, max_duration=300):
        """
        Create a highlight reel by concatenating individual highlights
        """
        if not self.highlights:
            print("No highlights available to create reel")
            return None
            
        # Sort highlights by importance/score
        sorted_highlights = self.sort_highlights_by_importance()
        
        # Select highlights that fit within max duration
        selected_clips = []
        total_duration = 0
        
        for highlight in sorted_highlights:
            clip_duration = highlight['end_time'] - highlight['start_time']
            if total_duration + clip_duration <= max_duration:
                selected_clips.append(highlight['clip'])
                total_duration += clip_duration
            else:
                break
                
        if not selected_clips:
            print("No clips selected for highlight reel")
            return None
            
        try:
            # Concatenate clips
            final_reel = concatenate_videoclips(selected_clips)
            
            # Write to file
            final_reel.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True
            )
            
            print(f"Highlight reel created: {output_path}")
            print(f"Duration: {total_duration:.1f} seconds")
            print(f"Clips included: {len(selected_clips)}")
            
            return output_path
            
        except Exception as e:
            print(f"Error creating highlight reel: {e}")
            return None
            
    def sort_highlights_by_importance(self):
        """Sort highlights by importance/excitement level"""
        def get_importance_score(highlight):
            event_type = highlight['event_type']
            metadata = highlight['metadata']
            
            # Base scores for different event types
            base_scores = {
                'combo': 8,
                'friendly_fire': 7,
                'punch': 5,
                'stratagem_usage': 6,
                'objective_update': 7,
                'squad_synergy': 4,
                'defense': 3
            }
            
            score = base_scores.get(event_type, 3)
            
            # Adjust score based on metadata
            if event_type == 'combo':
                combo_length = metadata.get('combo_length', 0)
                score += min(combo_length * 2, 10)
                
            if event_type == 'punch' and metadata.get('landed', False):
                score += 3
                
            if event_type == 'squad_synergy':
                synergy_score = metadata.get('score', 0)
                score += synergy_score * 5
                
            return score
            
        return sorted(self.highlights, key=get_importance_score, reverse=True)
        
    def generate_tactical_suggestions(self, events, game_type):
        """Generate tactical suggestions based on analyzed events"""
        suggestions = []
        
        if game_type.lower() == 'helldivers_2':
            suggestions = self.generate_helldivers_suggestions(events)
        elif game_type.lower() == 'undisputed_boxing':
            suggestions = self.generate_boxing_suggestions(events)
            
        return suggestions
        
    def generate_helldivers_suggestions(self, events):
        """Generate tactical suggestions for Helldivers 2"""
        suggestions = []
        
        # Analyze friendly fire incidents
        ff_events = [e for e in events if e.get('type') == 'friendly_fire']
        if len(ff_events) > 3:
            suggestions.append({
                'category': 'Team Coordination',
                'priority': 'high',
                'suggestion': 'Excessive friendly fire detected. Improve communication and positioning.',
                'specific_advice': 'Use voice chat to call out targets and avoid clustering together.'
            })
            
        # Analyze stratagem usage
        stratagem_events = [e for e in events if e.get('type') == 'stratagem_usage']
        if len(stratagem_events) < 5:
            suggestions.append({
                'category': 'Resource Management',
                'priority': 'medium',
                'suggestion': 'Low stratagem usage detected. Utilize your tools more effectively.',
                'specific_advice': 'Don\'t save stratagems for "perfect" moments - use them proactively.'
            })
            
        # Analyze squad synergy
        synergy_events = [e for e in events if e.get('type') == 'squad_synergy']
        low_synergy = [e for e in synergy_events if e.get('score', 0) < 0.5]
        if len(low_synergy) > len(synergy_events) * 0.6:
            suggestions.append({
                'category': 'Squad Tactics',
                'priority': 'high',
                'suggestion': 'Poor squad coordination detected.',
                'specific_advice': 'Maintain better formation and support teammates more actively.'
            })
            
        return suggestions
        
    def generate_boxing_suggestions(self, events):
        """Generate tactical suggestions for boxing"""
        suggestions = []
        
        # Analyze punch accuracy
        punch_events = [e for e in events if e.get('type') == 'punch']
        if punch_events:
            landed = len([e for e in punch_events if e.get('landed', False)])
            accuracy = landed / len(punch_events)
            
            if accuracy < 0.4:
                suggestions.append({
                    'category': 'Accuracy',
                    'priority': 'high',
                    'suggestion': f'Low punch accuracy ({accuracy:.1%}). Focus on precision over volume.',
                    'specific_advice': 'Practice timing and distance management. Quality over quantity.'
                })
                
        # Analyze combo usage
        combo_events = [e for e in events if e.get('type') == 'combo']
        if len(combo_events) < 3:
            suggestions.append({
                'category': 'Offense',
                'priority': 'medium',
                'suggestion': 'Limited combination punching detected.',
                'specific_advice': 'Work on chaining 2-3 punch combinations for more effective offense.'
            })
            
        # Analyze defense
        defense_events = [e for e in events if e.get('type') == 'defense']
        head_movement_events = [e for e in events if e.get('type') == 'head_movement']
        
        if len(defense_events) < 5 and len(head_movement_events) < 10:
            suggestions.append({
                'category': 'Defense',
                'priority': 'high',
                'suggestion': 'Insufficient defensive activity detected.',
                'specific_advice': 'Increase head movement and active defense. Don\'t just stand and trade.'
            })
            
        return suggestions
        
    def export_highlights_summary(self, output_path):
        """Export a summary of highlights to JSON"""
        summary = {
            'total_highlights': len(self.highlights),
            'highlights': []
        }
        
        for i, highlight in enumerate(self.highlights):
            highlight_summary = {
                'index': i,
                'event_type': highlight['event_type'],
                'description': highlight['description'],
                'timestamp': highlight['timestamp'],
                'start_time': highlight['start_time'],
                'end_time': highlight['end_time'],
                'duration': highlight['end_time'] - highlight['start_time']
            }
            summary['highlights'].append(highlight_summary)
            
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=4)
            
        return f"Highlights summary exported to {output_path}"
        
    def cleanup(self):
        """Clean up resources"""
        if self.clip:
            self.clip.close()
            
        for highlight in self.highlights:
            if 'clip' in highlight and highlight['clip']:
                highlight['clip'].close()

if __name__ == '__main__':
    # Test the highlight generator
    # This would normally be called with real video and events
    
    sample_events = [
        {
            'type': 'combo',
            'timestamp': 30.5,
            'description': '3-punch combo landed',
            'combo_length': 3,
            'punches_landed': 2
        },
        {
            'type': 'punch',
            'timestamp': 45.2,
            'description': 'Right hook landed',
            'landed': True,
            'punch_type': 'right_hand'
        }
    ]
    
    # Note: This would require an actual video file to test
    # generator = HighlightGenerator('path/to/video.mp4')
    # highlights = generator.generate_highlights(sample_events, 'undisputed_boxing')
    # suggestions = generator.generate_tactical_suggestions(sample_events, 'undisputed_boxing')
    
    print("Highlight generator module loaded successfully")
    print("Sample tactical suggestions structure:")
    
    # Show example suggestions structure
    example_suggestions = [
        {
            'category': 'Accuracy',
            'priority': 'high',
            'suggestion': 'Low punch accuracy detected.',
            'specific_advice': 'Focus on timing and distance management.'
        }
    ]
    
    for suggestion in example_suggestions:
        print(f"Category: {suggestion['category']}")
        print(f"Priority: {suggestion['priority']}")
        print(f"Suggestion: {suggestion['suggestion']}")
        print(f"Advice: {suggestion['specific_advice']}")

