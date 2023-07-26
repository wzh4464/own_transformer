import torch
import torch.optim as optim
from main import Transformer
from main import device
import torch.nn as nn
import time

def test_transformer():
    # define hyperparameters
    src_vocab_size = 100
    tgt_vocab_size = 100
    d_model = 128
    num_heads = 4
    num_layers = 2
    d_ff = 512
    max_seq_length = 50
    dropout = 0.1

    # create model
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, device)

    # create input tensors
    src = torch.randint(low=0, high=src_vocab_size, size=(2, 10))
    tgt = torch.randint(low=0, high=tgt_vocab_size, size=(2, 12))

    # test forward pass
    output = model(src, tgt)
    print("Device: ", output.device)
    assert output.shape == torch.Size([2, 12, tgt_vocab_size])

    # test generate_mask function
    src_mask, tgt_mask = model.generate_mask(src, tgt)
    assert src_mask.shape == torch.Size([2, 1, 1, 10])
    assert tgt_mask.shape == torch.Size([2, 1, 12, 12])
    
if __name__ == "__main__":
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_length = 100
    dropout = 0.1

    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, device=device)

    # Generate random sample data
    src_data = torch.randint(1, src_vocab_size, (64, max_seq_length)).to(device)  # (batch_size, seq_length)
    tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length)).to(device)  # (batch_size, seq_length)
    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
    optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    transformer.train()



for epoch in range(100):
    start_time = time.time()
    optimizer.zero_grad()
    output = transformer(src_data, tgt_data[:, :-1])
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
    loss.backward()
    optimizer.step()
    end_time = time.time()
    epoch_time = end_time - start_time
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}, Time: {epoch_time:.2f} seconds")