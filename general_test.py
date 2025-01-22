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

def test_hip_graph_memory_allocation():
    """Measure memory usage during HIP graph execution."""
    print("\n[HIP Graph Memory Allocation Test]")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This test requires a GPU.")

    device = torch.device("cuda")

    # Define a large matrix multiplication task
    a = torch.randn(4096, 4096, device=device)
    b = torch.randn(4096, 4096, device=device)
    result = torch.empty_like(a)
    graph = torch.cuda.CUDAGraph()

    # Capture the graph
    with torch.cuda.graph(graph):
        result = torch.matmul(a, b)

    # Non-graph memory usage
    def non_graph_workload():
        return torch.matmul(a, b)

    memory_non_graph = memory_usage_check(non_graph_workload)

    # Graph memory usage
    def hip_graph_workload():
        graph.replay()

    memory_graph = memory_usage_check(hip_graph_workload)

    print(f"Non-graph peak memory: {memory_non_graph:.2f} MB")
    print(f"HIP graph peak memory: {memory_graph:.2f} MB")

def test_hip_graph_operations_per_second():
    """Measure operations per second for a heavy workload using HIP graphs."""
    print("\n[HIP Graph Operations Per Second Test]")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This test requires a GPU.")

    device = torch.device("cuda")

    # Define a workload
    a = torch.randn(4096, 4096, device=device)
    b = torch.randn(4096, 4096, device=device)
    result = torch.empty_like(a)
    graph = torch.cuda.CUDAGraph()

    # Capture the graph
    with torch.cuda.graph(graph):
        result = torch.matmul(a, b)

    def non_graph_workload():
        return torch.matmul(a, b)

    def hip_graph_workload():
        graph.replay()

    # Benchmark throughput
    time_non_graph = benchmark_function(non_graph_workload)
    time_graph = benchmark_function(hip_graph_workload)

    ops = 2 * (4096 ** 3)  # Approx FLOPs for matmul
    print(f"Non-graph operations per second: {ops / (time_non_graph + 0.000001):.2e} ops/sec")
    print(f"HIP graph operations per second: {ops / (time_graph + 0.000001):.2e} ops/sec")

def test_mobilenet_training():
    """Train a MobileNet model over 50 epochs using HIP graphs."""
    print("\n[MobileNet Training with HIP Graphs]")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This test requires a GPU.")

    device = torch.device("cuda")

    # Load MobileNet model
    model = mobilenet_v2(weights=None).to(device)

    # Define dataset and data loader
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    dataset = datasets.FakeData(transform=transform, size=1000)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    static_inputs = torch.randn(32, 3, 128, 128, device=device)
    static_labels = torch.randint(0, 1000, (32,), device=device)
    graph = torch.cuda.CUDAGraph()

    # Capture the graph
    with torch.cuda.graph(graph):
        outputs = model(static_inputs)
        loss = criterion(outputs, static_labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Training loop
    for epoch in range(5):  # Use 5 epochs for demonstration
        epoch_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Use HIP graph
            static_inputs.copy_(inputs)
            static_labels.copy_(labels)
            graph.replay()

            # Accumulate loss
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/5, Loss: {epoch_loss:.4f}")

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

        # Test HIP graph memory allocation
        test_hip_graph_memory_allocation()

        # Test HIP graph operations per second
        test_hip_graph_operations_per_second()

        # Train MobileNet using HIP graphs
        test_mobilenet_training()

        # Memory transfer benchmark
        test_memory_transfer()

    except RuntimeError as e:
        print(f"Runtime Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
