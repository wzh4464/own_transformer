import torch
from main import MultiHeadAttention

# Test case 1
batch_size = 2
num_queries = 3
num_keys = 4
d_model = 8
num_heads = 2

Q = torch.randn(batch_size, num_queries, d_model)
K = torch.randn(batch_size, num_keys, d_model)
V = torch.randn(batch_size, num_keys, d_model)

multi_head_attn = MultiHeadAttention(d_model, num_heads)
output = multi_head_attn.scaled_dot_product_attention(Q, K, V)
assert output.shape == (batch_size, num_queries, d_model)

# Test case 2
batch_size = 1
num_queries = 5
num_keys = 5
d_model = 16
num_heads = 4

Q = torch.randn(batch_size, num_queries, d_model)
K = torch.randn(batch_size, num_keys, d_model)
V = torch.randn(batch_size, num_keys, d_model)

multi_head_attn = MultiHeadAttention(d_model, num_heads)
output = multi_head_attn.scaled_dot_product_attention(Q, K, V)
assert output.shape == (batch_size, num_queries, d_model)

# Test case 3
batch_size = 3
num_queries = 2
num_keys = 2
d_model = 32
num_heads = 8

Q = torch.randn(batch_size, num_queries, d_model)
K = torch.randn(batch_size, num_keys, d_model)
V = torch.randn(batch_size, num_keys, d_model)

multi_head_attn = MultiHeadAttention(d_model, num_heads)
output = multi_head_attn.scaled_dot_product_attention(Q, K, V)
assert output.shape == (batch_size, num_queries, d_model)