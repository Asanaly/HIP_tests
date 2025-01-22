import torch
import time
import numpy as np
from typing import Dict, Any, Tuple
import json

class HIPGraphTester:
    def __init__(self):
        # Verify we're running on ROCm
        if not torch.cuda.is_available():
            raise RuntimeError("No GPU available")
            
        self.device = torch.device('cuda')
        print(f"Using device: {torch.cuda.get_device_name()}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"ROCm support: {hasattr(torch.cuda, 'hipCompileAndLoad')}")
        
    def measure_performance(self, func, warmup=10, iterations=100) -> Dict[str, float]:
        """Measure execution time and memory usage"""
        # Warmup
        for _ in range(warmup):
            func()
            
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Measure time
        times = []
        peak_memory = 0
        
        for _ in range(iterations):
            torch.cuda.reset_peak_memory_stats()
            start_event.record()
            func()
            end_event.record()
            end_event.synchronize()
            times.append(start_event.elapsed_time(end_event))
            peak_memory = max(peak_memory, torch.cuda.max_memory_allocated())
            
        return {
            'avg_time_ms': np.mean(times),
            'std_time_ms': np.std(times),
            'peak_memory_mb': peak_memory / 1024 / 1024
        }
        
    def test_conv_graph(self, batch_size: int = 32, input_size: int = 224) -> Dict[str, Any]:
        """Test convolution operations with and without graph capture"""
        results = {}
        
        # Create sample input and model
        input_tensor = torch.randn(batch_size, 3, input_size, input_size, device=self.device)
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        ).to(self.device)
        
        # Baseline without graph
        baseline_func = lambda: model(input_tensor)
        results['baseline'] = self.measure_performance(baseline_func)
        
        # With graph capture
        g = torch.cuda.CUDAGraph()
        
        # Warmup before capture
        static_output = model(input_tensor)
        
        # Capture graph
        with torch.cuda.graph(g):
            graph_output = model(input_tensor)
            
        # Test graph replay
        graph_func = lambda: g.replay()
        results['graph'] = self.measure_performance(graph_func)
        
        return results
        
    def test_deep_network_graph(self, batch_size: int = 32) -> Dict[str, Any]:
        """Test deeper network with multiple operations"""
        results = {}
        
        # Create deeper model
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(256, 1000)
        ).to(self.device)
        
        input_tensor = torch.randn(batch_size, 3, 224, 224, device=self.device)
        
        # Baseline without graph
        baseline_func = lambda: model(input_tensor)
        results['baseline'] = self.measure_performance(baseline_func)
        
        # With graph capture
        g = torch.cuda.CUDAGraph()
        
        # Warmup before capture
        static_output = model(input_tensor)
        
        # Capture graph
        with torch.cuda.graph(g):
            graph_output = model(input_tensor)
            
        # Test graph replay
        graph_func = lambda: g.replay()
        results['graph'] = self.measure_performance(graph_func)
        
        return results

def run_all_tests():
    """Run comprehensive tests with different configurations"""
    tester = HIPGraphTester()
    all_results = {}
    
    # Test different batch sizes
    batch_sizes = [1, 8, 32, 64]
    input_sizes = [224, 416, 608]
    
    for batch_size in batch_sizes:
        for input_size in input_sizes:
            print(f"\nTesting batch size {batch_size}, input size {input_size}")
            
            # Simple conv test
            results = tester.test_conv_graph(batch_size, input_size)
            all_results[f'conv_b{batch_size}_s{input_size}'] = results
            
            # Deep network test
            if input_size == 224:  # Only run deep network on standard image size
                results = tester.test_deep_network_graph(batch_size)
                all_results[f'deep_b{batch_size}'] = results
                
    # Save results
    with open('hip_graph_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
        
    return all_results

if __name__ == "__main__":
    results = run_all_tests()
    
    # Print summary
    print("\nResults Summary:")
    for test_name, test_results in results.items():
        print(f"\n{test_name}:")
        print("Baseline:", test_results['baseline'])
        print("Graph:", test_results['graph'])