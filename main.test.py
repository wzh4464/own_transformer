import torch
from main import MultiHeadAttention, PositionWiseFeedForward, PositionalEncoding, EncoderLayer

# Test MultiHeadAttention
batch_size = 2
seq_length = 3
d_model = 4
num_heads = 2
d_k = d_v = d_model // num_heads

Q = torch.randn(batch_size, seq_length, d_model)
K = torch.randn(batch_size, seq_length, d_model)
V = torch.randn(batch_size, seq_length, d_model)
mask = torch.ones(batch_size, seq_length, seq_length)
attention = MultiHeadAttention(d_model, num_heads)
output = attention(Q, K, V, mask)
assert output.shape == (batch_size, seq_length, d_model)

# Test PositionWiseFeedForward
d_ff = 8
feed_forward = PositionWiseFeedForward(d_model, d_ff)
output = feed_forward(torch.randn(batch_size, seq_length, d_model))
assert output.shape == (batch_size, seq_length, d_model)

# Test PositionalEncoding
max_seq_length = 5
positional_encoding = PositionalEncoding(d_model, max_seq_length)
output = positional_encoding(torch.randn(batch_size, seq_length, d_model))
assert output.shape == (batch_size, seq_length, d_model)

# Test EncoderLayer
num_heads = 2
d_ff = 8
dropout = 0.1
encoder_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
output = encoder_layer(torch.randn(batch_size, seq_length, d_model), mask)
assert output.shape == (batch_size, seq_length, d_model)
