import torch
import torch.nn as nn
import torch.optim as optim
import time

# Define a very simple model with one linear layer
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(3, 2)  # Input of size 3, output of size 2

    def forward(self, x):
        return self.linear(x)

def simple_cuda_graph_test():
    # Ensure you're using GPU (CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA device not available. This test requires a GPU.")

    # Initialize model, inputs, labels, loss, and optimizer
    model = SimpleModel().to(device)
    inputs = torch.randn(1, 3, device=device)  # Random input tensor of size (1, 3)
    labels = torch.randn(1, 2, device=device)  # Random labels for comparison (output size is 2)

    criterion = nn.MSELoss()  # Mean Squared Error loss for simplicity
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Warm-up the model
    optimizer.zero_grad(set_to_none=True)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # Capture the graph
    print("Capturing CUDA Graph...")
    graph_capture_start = time.time()
    
    graph = torch.cuda.CUDAGraph()

    # Ensure optimizer state is static
    optimizer.zero_grad(set_to_none=True)

    # Capture the graph
    with torch.cuda.graph(graph):
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    graph_capture_end = time.time()
    graph_capture_time = graph_capture_end - graph_capture_start
    print(f"Graph capture time: {graph_capture_time:.4f} seconds")

    # Execute the graph multiple times for profiling
    print("Executing CUDA Graph...")
    exec_start = time.time()

    num_replays = 5  # Number of graph replays
    for _ in range(num_replays):
        graph.replay()

    exec_end = time.time()
    exec_time_total = exec_end - exec_start
    exec_time_avg = exec_time_total / num_replays

    print("\n--- Metrics ---")
    print(f"Final Loss: {loss.item():.4f}")
    print(f"Total Execution Time: {exec_time_total:.4f} seconds")
    print(f"Average Execution Time: {exec_time_avg:.4f} seconds")

# Run the test
if __name__ == "__main__":
    simple_cuda_graph_test()
