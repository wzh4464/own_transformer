# -*- coding: utf-8 -*-
# @Author: WU Zihan
# @Date:   2023-07-20 16:27:39
# @Last Modified by:   WU Zihan
# @Last Modified time: 2023-07-22 13:21:48
import torch

from main import MultiHeadAttention, PositionWiseFeedForward

def test_multi_head_attention():
    # create a multi-head attention module
    d_model = 64
    num_heads = 8
    mha = MultiHeadAttention(d_model, num_heads)

    # create some input tensors
    batch_size = 16
    seq_length = 10
    num_queries = 5
    num_keys = 7
    num_values = 7
    Q = torch.randn(batch_size, num_queries, d_model)
    K = torch.randn(batch_size, num_keys, d_model)
    V = torch.randn(batch_size, num_values, d_model)

    # test the forward pass
    output = mha(Q, K, V)
    assert output.shape == (batch_size, num_queries, d_model)

def test_position_wise_feed_forward():
    # create a position-wise feed-forward module
    d_model = 3
    d_ff = 4
    pwff = PositionWiseFeedForward(d_model, d_ff)

    # create some input tensors
    batch_size = 2
    seq_length = 3
    x = torch.randn(batch_size, seq_length, d_model)
    print("x:", x)

    # check if CUDA is available
    if torch.cuda.is_available():
        # move tensors to GPU
        x = x.cuda()
        pwff.cuda()

    # test the forward pass
    output = pwff(x)
    print("output:", output)
    assert output.shape == (batch_size, seq_length, d_model)
    
if __name__ == "__main__":
    test_position_wise_feed_forward()
    