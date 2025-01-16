import torch
import torch.nn as nn
import torch.optim as optim

# Define a very simple model
class SimpleConvModel(nn.Module):
    def __init__(self):
        super(SimpleConvModel, self).__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1)  # Single conv layer

    def forward(self, x):
        return self.conv(x)


def test_simple_conv_model_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA device not available. This test requires a GPU.")

    # Initialize model, inputs, labels, loss, and optimizer
    model = SimpleConvModel().to(device)
    static_inputs = torch.randn(1, 3, 32, 32, device=device)  # Fixed input size
    static_labels = torch.randn(1, 1, 32, 32, device=device)  # Fixed label size

    criterion = nn.MSELoss()  # Mean Squared Error loss for simplicity
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Warm-up iterations
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(static_inputs)
        loss = criterion(outputs, static_labels)
        loss.backward()
        optimizer.step()

    # Static output and loss tensors for graph execution
    static_outputs = torch.empty_like(static_labels, device=device)

    print("Capturing CUDA Graph...")
    graph = torch.cuda.CUDAGraph()

    # Ensure optimizer state is static
    optimizer.zero_grad(set_to_none=True)

    # Capture the graph
    with torch.cuda.graph(graph):
        static_outputs = model(static_inputs)
        loss = criterion(static_outputs, static_labels)
        loss.backward()
        optimizer.step()

    print("CUDA Graph captured and executed successfully.")
    print("Outputs mean:", static_outputs.mean().item())


# Run the test
if __name__ == "__main__":
    test_simple_conv_model_training()
