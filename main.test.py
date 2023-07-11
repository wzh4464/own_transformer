import torch

from main import MultiHeadAttention

# Test case 2
batch_size = 2
seq_length = 3
d_model = 4
num_heads = 2

model = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

x = torch.randn(batch_size, seq_length, d_model).cuda()

output = model.combine_heads(model.split_heads(x))
# assert x.values == output.values, "Test case 2 failed."
assert torch.all(torch.eq(x, output)), "Test case 2 failed."