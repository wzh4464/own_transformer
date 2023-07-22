# -*- coding: utf-8 -*-
# @Author: WU Zihan
# @Date:   2023-07-20 16:27:39
# @Last Modified by:   WU Zihan
# @Last Modified time: 2023-07-22 13:24:15
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

# multi-head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        """
        Multi-Head Attention module.
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        if torch.cuda.is_available():
            self.W_q = nn.Linear(d_model, d_model).cuda()
            self.W_k = nn.Linear(d_model, d_model).cuda()
            self.W_v = nn.Linear(d_model, d_model).cuda()
            self.W_o = nn.Linear(d_model, d_model).cuda()
            print("CUDA is available. Moving model to GPU.")
        else:
            # if mps is available, use mps
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            self.W_q = nn.Linear(d_model, d_model).to(device)
            self.W_k = nn.Linear(d_model, d_model).to(device)
            self.W_v = nn.Linear(d_model, d_model).to(device)
            self.W_o = nn.Linear(d_model, d_model).to(device)
            print(f"Using device: {device}")

    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        """
        Scaled Dot-Product Attention mechanism.
        
        Args:
            Q (torch.Tensor): Query tensor of shape (batch_size, num_queries, d_model).
            K (torch.Tensor): Key tensor of shape (batch_size, num_keys, d_model).
            V (torch.Tensor): Value tensor of shape (batch_size, num_values, d_model).
            mask (torch.Tensor, optional): Mask tensor of shape (batch_size, num_queries, num_keys). Defaults to None.
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_queries, d_model).
        """
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        """
        Split the last dimension of the input tensor into (num_heads, d_k).
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_model).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_heads, seq_length, d_k).
        """
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x):
        """
        Combine the (num_heads, d_k) dimensions of the input tensor into a single dimension.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_heads, seq_length, d_k).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, d_model).
        """
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
    
    def forward(self, Q, K, V, mask=None):
        """
        Forward pass of the Multi-Head Attention module.
        
        Args:
            Q (torch.Tensor): Query tensor of shape (batch_size, num_queries, d_model).
            K (torch.Tensor): Key tensor of shape (batch_size, num_keys, d_model).
            V (torch.Tensor): Value tensor of shape (batch_size, num_values, d_model).
            mask (torch.Tensor, optional): Mask tensor of shape (batch_size, num_queries, num_keys). Defaults to None.
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_queries, d_model).
        """
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output
        


if __name__ == "__main__":
    # test
    print(f"PyTorch version: {torch.__version__}")

    # Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
    print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
    print(f"Is MPS available? {torch.backends.mps.is_available()}")

    # Set the device      
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")