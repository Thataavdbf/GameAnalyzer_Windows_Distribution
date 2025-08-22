import json
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

class StatisticsEngine:
    def __init__(self):
        self.events = []
        self.game_type = None
        
    def load_events(self, events_data, game_type):
        """Load events data for analysis"""
        self.events = events_data
        self.game_type = game_type.lower()
        
    def generate_statistical_overlays(self):
        """Generate statistical overlays based on game type"""
        if self.game_type == "helldivers_2":
            return self.generate_helldivers_stats()
        elif self.game_type == "undisputed_boxing":
            return self.generate_boxing_stats()
        else:
            return self.generate_generic_stats()
            
    def generate_helldivers_stats(self):
        """Generate Helldivers 2 specific statistics"""
        stats = {
            'friendly_fire_analysis': self.analyze_friendly_fire(),
            'stratagem_efficiency': self.analyze_stratagem_usage(),
            'squad_performance': self.analyze_squad_performance(),
            'objective_completion': self.analyze_objectives(),
            'survival_analysis': self.analyze_survival_patterns()
        }
        return stats
        
    def generate_boxing_stats(self):
        """Generate Undisputed Boxing specific statistics"""
        stats = {
            'punch_analysis': self.analyze_punch_patterns(),
            'combo_effectiveness': self.analyze_combo_patterns(),
            'defense_rating': self.analyze_defense_patterns(),
            'stamina_management': self.analyze_stamina_patterns(),
            'fight_rhythm': self.analyze_fight_rhythm()
        }
        return stats
        
    def analyze_friendly_fire(self):
        """Analyze friendly fire incidents in Helldivers 2"""
        ff_events = [e for e in self.events if e.get('type') == 'friendly_fire']
        
        if not ff_events:
            return {'incidents': 0, 'rate': 0, 'severity': 'none'}
            
        total_incidents = len(ff_events)
        
        # Calculate incidents per minute
        if self.events:
            duration = max([e.get('timestamp', 0) for e in self.events])
            incidents_per_minute = total_incidents / (duration / 60) if duration > 0 else 0
        else:
            incidents_per_minute = 0
            
        # Determine severity
        if incidents_per_minute > 2:
            severity = 'critical'
        elif incidents_per_minute > 1:
            severity = 'high'
        elif incidents_per_minute > 0.5:
            severity = 'moderate'
        else:
            severity = 'low'
            
        return {
            'incidents': total_incidents,
            'rate': incidents_per_minute,
            'severity': severity,
            'recommendation': self.get_ff_recommendation(severity)
        }
        
    def analyze_stratagem_usage(self):
        """Analyze stratagem usage patterns"""
        stratagem_events = [e for e in self.events if e.get('type') == 'stratagem_usage']
        
        if not stratagem_events:
            return {'total_used': 0, 'efficiency': 0, 'most_used': None}
            
        stratagem_types = [e.get('stratagem_type', 'unknown') for e in stratagem_events]
        stratagem_counter = Counter(stratagem_types)
        
        total_used = len(stratagem_events)
        most_used = stratagem_counter.most_common(1)[0] if stratagem_counter else None
        
        # Calculate efficiency based on timing and context
        efficiency_score = self.calculate_stratagem_efficiency(stratagem_events)
        
        return {
            'total_used': total_used,
            'efficiency': efficiency_score,
            'most_used': most_used[0] if most_used else None,
            'usage_distribution': dict(stratagem_counter),
            'recommendation': self.get_stratagem_recommendation(efficiency_score)
        }
        
    def analyze_squad_performance(self):
        """Analyze squad synergy and performance"""
        squad_events = [e for e in self.events if e.get('type') == 'squad_synergy']
        
        if not squad_events:
            return {'synergy_score': 0, 'coordination': 'poor'}
            
        scores = [e.get('score', 0) for e in squad_events]
        avg_score = np.mean(scores) if scores else 0
        
        if avg_score > 0.8:
            coordination = 'excellent'
        elif avg_score > 0.6:
            coordination = 'good'
        elif avg_score > 0.4:
            coordination = 'fair'
        else:
            coordination = 'poor'
            
        return {
            'synergy_score': avg_score,
            'coordination': coordination,
            'team_events': len(squad_events),
            'recommendation': self.get_squad_recommendation(coordination)
        }
        
    def analyze_objectives(self):
        """Analyze objective completion patterns"""
        obj_events = [e for e in self.events if e.get('type') == 'objective_update']
        
        completed = len([e for e in obj_events if e.get('status') == 'completed'])
        in_progress = len([e for e in obj_events if e.get('status') == 'in_progress'])
        
        completion_rate = completed / len(obj_events) if obj_events else 0
        
        return {
            'completed': completed,
            'in_progress': in_progress,
            'completion_rate': completion_rate,
            'efficiency': 'high' if completion_rate > 0.7 else 'medium' if completion_rate > 0.4 else 'low'
        }
        
    def analyze_survival_patterns(self):
        """Analyze survival and death patterns"""
        # This would analyze death events, revival patterns, etc.
        # Simplified for now
        return {
            'survival_time': 'N/A',
            'death_causes': {},
            'revival_efficiency': 'N/A'
        }
        
    def analyze_punch_patterns(self):
        """Analyze punching patterns in boxing"""
        punch_events = [e for e in self.events if e.get('type') == 'punch']
        
        if not punch_events:
            return {'total_punches': 0, 'accuracy': 0, 'power_distribution': {}}
            
        total_punches = len(punch_events)
        landed_punches = len([e for e in punch_events if e.get('landed', False)])
        accuracy = landed_punches / total_punches if total_punches > 0 else 0
        
        # Analyze punch types
        punch_types = [e.get('punch_type', 'unknown') for e in punch_events]
        type_distribution = Counter(punch_types)
        
        return {
            'total_punches': total_punches,
            'landed_punches': landed_punches,
            'accuracy': accuracy,
            'punch_types': dict(type_distribution),
            'recommendation': self.get_punch_recommendation(accuracy)
        }
        
    def analyze_combo_patterns(self):
        """Analyze combination punching patterns"""
        combo_events = [e for e in self.events if e.get('type') == 'combo']
        
        if not combo_events:
            return {'total_combos': 0, 'avg_length': 0, 'effectiveness': 0}
            
        total_combos = len(combo_events)
        combo_lengths = [e.get('combo_length', 0) for e in combo_events]
        avg_length = np.mean(combo_lengths) if combo_lengths else 0
        
        # Calculate effectiveness based on landed punches in combos
        total_combo_punches = sum([e.get('punches_landed', 0) for e in combo_events])
        total_combo_attempts = sum(combo_lengths)
        effectiveness = total_combo_punches / total_combo_attempts if total_combo_attempts > 0 else 0
        
        return {
            'total_combos': total_combos,
            'avg_length': avg_length,
            'effectiveness': effectiveness,
            'recommendation': self.get_combo_recommendation(effectiveness)
        }
        
    def analyze_defense_patterns(self):
        """Analyze defensive patterns"""
        defense_events = [e for e in self.events if e.get('type') == 'defense']
        head_movement_events = [e for e in self.events if e.get('type') == 'head_movement']
        
        defense_score = 0
        if defense_events:
            defense_scores = [e.get('effectiveness', 0) for e in defense_events]
            defense_score = np.mean(defense_scores)
            
        head_movement_score = 0
        if head_movement_events:
            movement_scores = [e.get('movement_score', 0) for e in head_movement_events]
            head_movement_score = np.mean(movement_scores)
            
        overall_defense = (defense_score + head_movement_score) / 2
        
        return {
            'defense_score': defense_score,
            'head_movement_score': head_movement_score,
            'overall_defense': overall_defense,
            'recommendation': self.get_defense_recommendation(overall_defense)
        }
        
    def analyze_stamina_patterns(self):
        """Analyze stamina management"""
        stamina_events = [e for e in self.events if e.get('type') == 'stamina']
        
        if not stamina_events:
            return {'management': 'unknown', 'fatigue_periods': 0}
            
        stamina_levels = [e.get('stamina_level', 'medium') for e in stamina_events]
        low_stamina_count = stamina_levels.count('low')
        
        management_quality = 'good' if low_stamina_count < len(stamina_levels) * 0.3 else 'poor'
        
        return {
            'management': management_quality,
            'fatigue_periods': low_stamina_count,
            'stamina_distribution': Counter(stamina_levels),
            'recommendation': self.get_stamina_recommendation(management_quality)
        }
        
    def analyze_fight_rhythm(self):
        """Analyze fight rhythm and pacing"""
        punch_events = [e for e in self.events if e.get('type') == 'punch']
        
        if len(punch_events) < 2:
            return {'rhythm': 'insufficient_data', 'pace': 'unknown'}
            
        # Calculate time intervals between punches
        timestamps = [e.get('timestamp', 0) for e in punch_events]
        intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        
        avg_interval = np.mean(intervals) if intervals else 0
        rhythm_consistency = 1 / (np.std(intervals) + 0.1) if intervals else 0
        
        if avg_interval < 1:
            pace = 'aggressive'
        elif avg_interval < 3:
            pace = 'moderate'
        else:
            pace = 'conservative'
            
        return {
            'rhythm_consistency': rhythm_consistency,
            'pace': pace,
            'avg_punch_interval': avg_interval,
            'recommendation': self.get_rhythm_recommendation(pace, rhythm_consistency)
        }
        
    def generate_generic_stats(self):
        """Generate generic statistics for unknown game types"""
        event_types = Counter([e.get('type', 'unknown') for e in self.events])
        
        return {
            'total_events': len(self.events),
            'event_types': dict(event_types),
            'duration': max([e.get('timestamp', 0) for e in self.events]) if self.events else 0
        }
        
    def calculate_stratagem_efficiency(self, stratagem_events):
        """Calculate stratagem usage efficiency"""
        # This would analyze timing, context, and effectiveness
        # Simplified for now
        return 0.75  # Placeholder
        
    def get_ff_recommendation(self, severity):
        """Get friendly fire recommendations"""
        recommendations = {
            'critical': "CRITICAL: Review team communication and positioning immediately",
            'high': "Focus on target identification and communication",
            'moderate': "Be more careful with area-of-effect weapons",
            'low': "Good friendly fire discipline, keep it up"
        }
        return recommendations.get(severity, "Monitor friendly fire incidents")
        
    def get_stratagem_recommendation(self, efficiency):
        """Get stratagem usage recommendations"""
        if efficiency > 0.8:
            return "Excellent stratagem usage"
        elif efficiency > 0.6:
            return "Good stratagem timing, consider more variety"
        else:
            return "Work on stratagem timing and selection"
            
    def get_squad_recommendation(self, coordination):
        """Get squad coordination recommendations"""
        recommendations = {
            'excellent': "Outstanding teamwork, maintain this level",
            'good': "Good coordination, work on tighter formations",
            'fair': "Improve communication and positioning",
            'poor': "Focus on basic team coordination drills"
        }
        return recommendations.get(coordination, "Work on team coordination")
        
    def get_punch_recommendation(self, accuracy):
        """Get punching recommendations"""
        if accuracy > 0.7:
            return "Excellent accuracy, focus on power and combinations"
        elif accuracy > 0.5:
            return "Good accuracy, work on timing and precision"
        else:
            return "Focus on accuracy over volume"
            
    def get_combo_recommendation(self, effectiveness):
        """Get combo recommendations"""
        if effectiveness > 0.7:
            return "Great combo work, try more complex combinations"
        elif effectiveness > 0.5:
            return "Good combos, work on follow-through"
        else:
            return "Practice basic 2-3 punch combinations"
            
    def get_defense_recommendation(self, defense_score):
        """Get defense recommendations"""
        if defense_score > 0.7:
            return "Solid defense, maintain head movement"
        elif defense_score > 0.5:
            return "Good defense, work on counter-attacking"
        else:
            return "Focus on basic defensive fundamentals"
            
    def get_stamina_recommendation(self, management):
        """Get stamina management recommendations"""
        if management == 'good':
            return "Good stamina management, maintain pace"
        else:
            return "Work on pacing and stamina conservation"
            
    def get_rhythm_recommendation(self, pace, consistency):
        """Get fight rhythm recommendations"""
        if pace == 'aggressive' and consistency > 0.5:
            return "Good aggressive rhythm, watch for counters"
        elif pace == 'conservative':
            return "Consider increasing output in key moments"
        else:
            return "Work on establishing a consistent rhythm"
            
    def export_statistics(self, filename):
        """Export statistics to JSON file"""
        stats = self.generate_statistical_overlays()
        with open(filename, 'w') as f:
            json.dump(stats, f, indent=4, default=str)
        return f"Statistics exported to {filename}"

if __name__ == '__main__':
    # Test the statistics engine
    engine = StatisticsEngine()
    
    # Sample events for testing
    sample_events = [
        {'type': 'punch', 'timestamp': 1.0, 'punch_type': 'left_hand', 'landed': True},
        {'type': 'punch', 'timestamp': 2.5, 'punch_type': 'right_hand', 'landed': False},
        {'type': 'combo', 'timestamp': 3.0, 'combo_length': 3, 'punches_landed': 2},
        {'type': 'defense', 'timestamp': 4.0, 'effectiveness': 0.8}
    ]
    
    engine.load_events(sample_events, 'undisputed_boxing')
    stats = engine.generate_statistical_overlays()
    
    print("Generated Statistics:")
    for category, data in stats.items():
        print(f"\n{category.upper()}:")
        for key, value in data.items():
            print(f"  {key}: {value}")

