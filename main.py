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
            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            self.W_o = nn.Linear(d_model, d_model)
            print("CUDA is not available. Using CPU.")

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