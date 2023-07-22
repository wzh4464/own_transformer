# -*- coding: utf-8 -*-
# @Author: WU Zihan
# @Date:   2023-07-20 16:27:39
# @Last Modified by:   WU Zihan
# @Last Modified time: 2023-07-22 13:21:48
import torch

from main import MultiHeadAttention

# Test case 2
batch_size = 2
seq_length = 3
d_model = 4
num_heads = 2
num_queries = 3
num_keys = 5
window_size = 3

# Create a mask tensor with shape (batch_size, num_queries, num_keys)
device = "mps" if torch.backends.mps.is_available() else "cuda"
mask = torch.zeros(batch_size, num_queries, num_keys).to(device)

# Set the mask values for non-neighboring positions to -inf
for i in range(seq_length):
    mask[:, i, :max(0, i - window_size)] = float('-inf')
    mask[:, i, i + window_size + 1:] = float('-inf')

Q = torch.rand(batch_size, num_queries, d_model).to(device)
K = torch.rand(batch_size, num_keys, d_model).to(device)
V = torch.rand(batch_size, num_keys, d_model).to(device)

# Pass the mask tensor to the Multi-Head Attention module
module = MultiHeadAttention(d_model, num_heads)
output = module.forward(Q, K, V, mask=mask)
assert output.shape == (batch_size, num_queries, d_model)