import torch
import math
from typing import List

import torch
from torch import nn
import numpy as np
import torch.cuda.profiler as profiler

model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True).cuda()

img_size=512
batch_size=8

input = torch.rand(batch_size, 3, img_size, img_size)
label = torch.ones(batch_size, 1, img_size, img_size)

# static_input a placeholder to take input for the graphed_model, \
# during training we need to copy real data into it. \
# So the training data will be used. Same for other static_*variables.

static_input = input.cuda()
static_target = label.cuda()

optimizer=torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)

loss_fn = torch.nn.BCEWithLogitsLoss()


iters = 10 # first 5 to warm up
iter_to_capture = 9

iters = 10 # first 5 to warm up
iter_to_capture = 9

s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())

# warmup
with torch.cuda.stream(s):
    for i in range(3):
        optimizer.zero_grad(set_to_none=True)
        y_pred = model(static_input)
        loss = loss_fn(y_pred, static_target)
        loss.backward()
        optimizer.step()
torch.cuda.current_stream().wait_stream(s)

graphed_model = torch.cuda.CUDAGraph()
# Sets grads to None before capture, so backward() will create
# .grad attributes with allocations from the graph's private pool
optimizer.zero_grad(set_to_none=True)
with torch.cuda.graph(graphed_model):
    # the graph capturing starts, as you can see we go through the full
    # training iterations here:
    static_y_pred = model(static_input) # Forward pass
    static_loss = loss_fn(static_y_pred, static_target) # Loss computation
    static_loss.backward() # Backward pass
    optimizer.step() # Optimization step

with torch.autograd.profiler.emit_nvtx(): # For profiling purpose
    for iter in range(iters):
        static_input.copy_(input)
        static_target.copy_(label)
        if iter == iter_to_capture: # For profiling purpose
            profiler.start() # For profiling purpose
        # instead of using the original model for training
        # here we use the graphed model and just replay those kernels.
        graphed_model.replay()
        print(static_loss.item()) # loss should go down
        profiler.stop() # For profiling purpose