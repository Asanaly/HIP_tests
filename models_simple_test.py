import torch
import time
from torch import nn

# Load the model
model = torch.hub.load(
    'mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=1, init_features=32, pretrained=True
).cuda()

# Define input size, batch size, and dummy data
img_size = 512
batch_size = 8
epochs = 5

# Dummy training data
input = torch.rand(batch_size, 3, img_size, img_size).cuda()
label = torch.ones(batch_size, 1, img_size, img_size).cuda()

# Loss function and optimizer
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(
    model.parameters(), lr=0.001, betas=(0.9, 0.999),
    eps=1e-08, weight_decay=0.01, amsgrad=False
)

# Measure graph initialization time
start_time = time.time()

# CUDA Graph setup
static_input = input.clone().cuda()
static_target = label.clone().cuda()
optimizer.zero_grad(set_to_none=True)

# Warm-up phase
for _ in range(3):
    y_pred = model(static_input)
    loss = loss_fn(y_pred, static_target)
    loss.backward()
    optimizer.step()

torch.cuda.synchronize()  # Ensure all GPU operations finish

# Capture graph
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    static_y_pred = model(static_input)
    static_loss = loss_fn(static_y_pred, static_target)
    static_loss.backward()
    optimizer.step()

graph_init_time = time.time() - start_time

# Training with captured graph
print("Starting Training...")
total_time = 0
final_loss = None

for epoch in range(epochs):
    start_epoch_time = time.time()

    for _ in range(10):  # Simulating 10 iterations per epoch
        static_input.copy_(input)
        static_target.copy_(label)

        graph.replay()  # Replay the graph

        # Capture the loss for the final batch
        final_loss = static_loss.item()

    torch.cuda.synchronize()
    epoch_time = time.time() - start_epoch_time
    total_time += epoch_time
    print(f"Epoch {epoch + 1}/{epochs} completed in {epoch_time:.4f}s")

# Calculate metrics
avg_epoch_time = total_time / epochs

print("\n--- Summary ---")
print(f"Number of Epochs: {epochs}")
print(f"Final Loss: {final_loss:.4f}")
print(f"Total Training Time: {total_time:.4f}s")
print(f"Average Epoch Time: {avg_epoch_time:.4f}s")
print(f"Graph Initialization Time: {graph_init_time:.4f}s")
