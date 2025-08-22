import time
import psutil
import cProfile
import pstats
from functools import wraps
import numpy as np

class PerformanceProfiler:
    def __init__(self):
        self.metrics = {}
        self.start_time = None
        self.memory_usage = []
        
    def start_profiling(self):
        """Start performance profiling"""
        self.start_time = time.time()
        self.memory_usage = []
        
    def record_memory_usage(self):
        """Record current memory usage"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.memory_usage.append(memory_mb)
        return memory_mb
        
    def time_function(self, func_name):
        """Decorator to time function execution"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start = time.time()
                result = func(*args, **kwargs)
                end = time.time()
                
                if func_name not in self.metrics:
                    self.metrics[func_name] = []
                self.metrics[func_name].append(end - start)
                
                return result
            return wrapper
        return decorator
        
    def profile_video_processing(self, video_processor, video_path):
        """Profile video processing performance"""
        print("Profiling video processing...")
        
        # Profile video loading
        start = time.time()
        processor = video_processor(video_path)
        load_time = time.time() - start
        
        # Profile frame extraction
        frame_times = []
        for i in range(0, min(10, int(processor.get_duration()))):
            start = time.time()
            frame = processor.extract_frame(i)
            frame_times.append(time.time() - start)
            
        processor.close()
        
        return {
            'video_load_time': load_time,
            'avg_frame_extraction_time': np.mean(frame_times),
            'max_frame_extraction_time': np.max(frame_times),
            'min_frame_extraction_time': np.min(frame_times)
        }
        
    def profile_analysis(self, analyzer, test_frames):
        """Profile analysis performance"""
        print("Profiling analysis performance...")
        
        analysis_times = []
        for i, frame in enumerate(test_frames):
            start = time.time()
            events = analyzer.analyze_frame(frame, i * 0.033)  # 30fps
            analysis_times.append(time.time() - start)
            
        return {
            'avg_analysis_time': np.mean(analysis_times),
            'max_analysis_time': np.max(analysis_times),
            'min_analysis_time': np.min(analysis_times),
            'total_frames_analyzed': len(test_frames)
        }
        
    def profile_memory_usage(self, func, *args, **kwargs):
        """Profile memory usage of a function"""
        process = psutil.Process()
        
        # Record initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Execute function
        start = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start
        
        # Record final memory
        final_memory = process.memory_info().rss / 1024 / 1024
        
        return {
            'result': result,
            'execution_time': execution_time,
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'memory_increase_mb': final_memory - initial_memory
        }
        
    def generate_performance_report(self):
        """Generate a comprehensive performance report"""
        report = {
            'total_runtime': time.time() - self.start_time if self.start_time else 0,
            'function_metrics': {},
            'memory_stats': {}
        }
        
        # Process function metrics
        for func_name, times in self.metrics.items():
            report['function_metrics'][func_name] = {
                'avg_time': np.mean(times),
                'max_time': np.max(times),
                'min_time': np.min(times),
                'total_calls': len(times),
                'total_time': np.sum(times)
            }
            
        # Process memory stats
        if self.memory_usage:
            report['memory_stats'] = {
                'avg_memory_mb': np.mean(self.memory_usage),
                'max_memory_mb': np.max(self.memory_usage),
                'min_memory_mb': np.min(self.memory_usage),
                'memory_samples': len(self.memory_usage)
            }
            
        return report
        
    def optimize_recommendations(self, report):
        """Generate optimization recommendations based on profiling results"""
        recommendations = []
        
        # Check for slow functions
        for func_name, metrics in report.get('function_metrics', {}).items():
            if metrics['avg_time'] > 0.1:  # Functions taking more than 100ms
                recommendations.append({
                    'type': 'performance',
                    'priority': 'high',
                    'function': func_name,
                    'issue': f'Function {func_name} is slow (avg: {metrics["avg_time"]:.3f}s)',
                    'suggestion': 'Consider optimizing this function or using caching'
                })
                
        # Check memory usage
        memory_stats = report.get('memory_stats', {})
        if memory_stats.get('max_memory_mb', 0) > 1000:  # More than 1GB
            recommendations.append({
                'type': 'memory',
                'priority': 'medium',
                'issue': f'High memory usage detected ({memory_stats["max_memory_mb"]:.1f}MB)',
                'suggestion': 'Consider processing video in smaller chunks or optimizing data structures'
            })
            
        # Check total runtime
        if report.get('total_runtime', 0) > 300:  # More than 5 minutes
            recommendations.append({
                'type': 'runtime',
                'priority': 'medium',
                'issue': f'Long total runtime ({report["total_runtime"]:.1f}s)',
                'suggestion': 'Consider parallel processing or reducing analysis frequency'
            })
            
        return recommendations
        
    def run_comprehensive_profile(self, video_processor, analyzer, video_path):
        """Run a comprehensive performance profile"""
        print("Starting comprehensive performance profiling...")
        
        self.start_profiling()
        
        # Profile video processing
        video_metrics = self.profile_video_processing(video_processor, video_path)
        
        # Create test frames for analysis profiling
        test_frames = []
        try:
            processor = video_processor(video_path)
            for i in range(5):  # Test with 5 frames
                frame = processor.extract_frame(i)
                if frame is not None:
                    test_frames.append(frame)
            processor.close()
        except Exception as e:
            print(f"Error creating test frames: {e}")
            test_frames = [np.zeros((480, 640, 3), dtype=np.uint8)]  # Fallback
            
        # Profile analysis
        analysis_metrics = self.profile_analysis(analyzer, test_frames)
        
        # Generate report
        report = self.generate_performance_report()
        report['video_processing'] = video_metrics
        report['analysis_processing'] = analysis_metrics
        
        # Generate recommendations
        recommendations = self.optimize_recommendations(report)
        report['recommendations'] = recommendations
        
        return report
        
    def save_profile_report(self, report, filename):
        """Save profiling report to file"""
        import json
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
            
        # Clean report for JSON serialization
        clean_report = {}
        for key, value in report.items():
            if isinstance(value, dict):
                clean_report[key] = {k: convert_numpy(v) for k, v in value.items()}
            else:
                clean_report[key] = convert_numpy(value)
                
        with open(filename, 'w') as f:
            json.dump(clean_report, f, indent=4, default=str)
            
        print(f"Performance report saved to {filename}")

def profile_function(profiler, func_name):
    """Decorator factory for profiling functions"""
    return profiler.time_function(func_name)

if __name__ == '__main__':
    # Test the profiler
    profiler = PerformanceProfiler()
    
    # Test timing a simple function
    @profiler.time_function('test_function')
    def test_function():
        time.sleep(0.1)
        return "test"
        
    # Run test function multiple times
    for _ in range(5):
        test_function()
        profiler.record_memory_usage()
        
    # Generate report
    report = profiler.generate_performance_report()
    print("Performance Report:")
    for key, value in report.items():
        print(f"{key}: {value}")
        
    # Generate recommendations
    recommendations = profiler.optimize_recommendations(report)
    print("\nOptimization Recommendations:")
    for rec in recommendations:
        print(f"- {rec['type'].upper()}: {rec['issue']}")
        print(f"  Suggestion: {rec['suggestion']}")

