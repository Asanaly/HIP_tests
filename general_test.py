import torch
import time
from torchvision.models import mobilenet_v2
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim

def benchmark_function(func, *args, **kwargs):
    """Benchmark a function's execution time."""
    start = time.time()
    func(*args, **kwargs)
    end = time.time()
    return end - start

def memory_usage_check(func, *args, **kwargs):
    """Measure memory usage during execution."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    func(*args, **kwargs)
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB

    return peak_memory

def validate_outputs(output1, output2, atol=1e-6):
    """Validate outputs between two methods."""
    if not torch.allclose(output1, output2, atol=atol):
        raise ValueError("Output mismatch detected!")
    print("Output validation passed.")

def test_hip_graph_execution():
    """Test HIP graph execution vs non-graph execution."""
    print("\n[HIP Graph Execution Test]")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This test requires a GPU.")

    device = torch.device("cuda")

    # Define a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(1024, 2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 1024)
    ).to(device)

    # Define inputs
    inputs = torch.randn(1024, 1024, device=device)

    # Non-graph execution
    def non_graph_func():
        return model(inputs)

    time_non_graph = benchmark_function(non_graph_func)

    # HIP graph execution
    static_inputs = inputs.clone()
    static_outputs = torch.empty_like(static_inputs)
    graph = torch.cuda.CUDAGraph()

    # Capture the graph
    with torch.cuda.graph(graph):
        static_outputs = model(static_inputs)

    # Replay the graph
    def hip_graph_func():
        graph.replay()

    time_graph = benchmark_function(hip_graph_func)

    print(f"Non-graph execution time: {time_non_graph:.4f} s")
    print(f"HIP graph execution time: {time_graph:.4f} s")

    # Check results for correctness
    validate_outputs(non_graph_func(), static_outputs)

def test_memory_transfer():
    """Benchmark CPU-to-GPU memory transfer."""
    print("\n[Memory Transfer Test]")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This test requires a GPU.")

    size = (1024, 1024)
    cpu_tensor = torch.randn(size)
    gpu_tensor = torch.empty(size, device="cuda")

    # Warm-up
    gpu_tensor.copy_(cpu_tensor)

    # Measure transfer time
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        gpu_tensor.copy_(cpu_tensor)
    torch.cuda.synchronize()
    end = time.perf_counter()

    print(f"Average transfer time: {(end - start) / 100:.6f} seconds")

if __name__ == "__main__":
    print("Running HIP Graph Tests")
    try:
        # Test HIP graph execution
        test_hip_graph_execution()

        # Memory transfer benchmark
        test_memory_transfer()

    except RuntimeError as e:
        print(f"Runtime Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
