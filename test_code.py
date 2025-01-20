import torch
import torch.nn as nn
import torch.optim as optim
import time

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

    # Normal training loop (no CUDA Graph)
    print("Training without CUDA Graph...")
    for _ in range(10):  # Multiple iterations for testing
        optimizer.zero_grad(set_to_none=True)
        static_outputs = model(static_inputs)
        loss = criterion(static_outputs, static_labels)
        loss.backward()
        optimizer.step()

    print(f"Final Loss: {loss.item():.4f}")
