import torch
from main import Transformer
from main import device

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
    test_transformer()