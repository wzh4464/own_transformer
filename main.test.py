import torch
from main import MultiHeadAttention, PositionWiseFeedForward, PositionalEncoding

# Test MultiHeadAttention
batch_size = 2
seq_length = 3
d_model = 4
num_heads = 2
d_k = d_v = d_model // num_heads

Q = torch.randn(batch_size, seq_length, d_model)
K = torch.randn(batch_size, seq_length, d_model)
V = torch.randn(batch_size, seq_length, d_model)

attn = MultiHeadAttention(d_model, num_heads)
output = attn(Q, K, V)
assert output.shape == (batch_size, seq_length, d_model)

# Test PositionWiseFeedForward
d_ff = 16
ffn = PositionWiseFeedForward(d_model, d_ff)
x = torch.randn(batch_size, seq_length, d_model)
output = ffn(x)
assert output.shape == (batch_size, seq_length, d_model)

# Test PositionalEncoding
max_seq_length = 5
pos_enc = PositionalEncoding(d_model, max_seq_length)
x = torch.randn(batch_size, seq_length, d_model)
output = pos_enc(x)
assert output.shape == (batch_size, seq_length, d_model)