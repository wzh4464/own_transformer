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

# Set the device      
# cpu / cuda / mps
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# multi-head attention
class MultiHeadAttention(nn.Module):
    '''
    Multi-Head Attention module.
    
    Args:
        d_model (int): Hidden dimension of the input tensor.
        num_heads (int): Number of attention heads.
        device (str, optional): Device to use.
        
    计算多头自注意力，使模型能够关注输入序列的某些不同方面
    '''
    def __init__(self, d_model: int, num_heads: int, device: str=device):
        """
        Multi-Head Attention module.
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.device = device
        
        self.W_q = nn.Linear(d_model, d_model).to(device)
        self.W_k = nn.Linear(d_model, d_model).to(device)
        self.W_v = nn.Linear(d_model, d_model).to(device)
        self.W_o = nn.Linear(d_model, d_model).to(device)
        # print(f"Using device: {device}")

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
            if mask.device.type != self.device:
                mask = mask.to(self.device)
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
        if Q.device.type != self.device:
            Q = Q.to(self.device)
        Q = self.split_heads(self.W_q(Q))
        if K.device.type != self.device:
            K = K.to(self.device)
        K = self.split_heads(self.W_k(K))
        if V.device.type != self.device:
            V = V.to(self.device)
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output
        
class PositionWiseFeedForward(nn.Module):
    """
    Position-Wise Feed-Forward module.
    
    Args:
        d_model (int): Hidden dimension of the input tensor.
        d_ff (int): Hidden dimension of the output tensor.
        device (str, optional): Device to use.
    此过程使模型能够在进行预测时考虑输入元素的位置.
    
    Transformer中的FFN全称是Position-wise Feed-Forward Networks，重点就是这个position-wise，区别于普通的全连接网络，这里FFN的输入是序列中每个位置上的元素，而不是整个序列，所以每个元素完全可以独立计算，最极端节省内存的做法是遍历序列，每次只取一个元素得到FFN的结果，但是这样做时间消耗太大，“分段”的含义就是做下折中，将序列分成段，也就是个子序列，每次读取一个子序列进行FFN计算，最后将份的结果拼接.分段FFN只是一种计算上的技巧，计算结果和原始FFN完全一致，所以不会影响到模型效果，好处是不需要一次性将整个序列读入内存，劣势当然是会增加额外的时间开销了.
    """
    def __init__(self, d_model: int, d_ff: int, device: str=device):
        super(PositionWiseFeedForward, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(d_model, d_ff).to(device)
        self.fc2 = nn.Linear(d_ff, d_model).to(device)
        self.activation = nn.ReLU().to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Position-Wise Feed-Forward module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_model).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, d_model).
        """
        # if x is not moved to device, move it
        if x.device.type != self.device:
            x = x.to(self.device)
        output = self.fc2(self.activation(self.fc1(x)))
        return output

class PositionalEncoding(nn.Module):
    """
    Positional Encoding module.
    
    $$
    PE_{(pos, 2i)} = sin(pos / 10000^{2i / d_{model}})
    PE_{(pos, 2i+1)} = cos(pos / 10000^{2i / d_{model}})
    $$
    
    Args:
        d_model (int): Hidden dimension of the input tensor.
        max_seq_length (int): Maximum sequence length.
        device (str, optional): Device to use.
        
    为输入序列中的每个位置添加一个可学习的向量，以便模型能够考虑序列中元素的顺序.
    """
    def __init__(self, d_model : int, max_seq_length : int, device: str=device):
        super(PositionalEncoding, self).__init__()
        
        self.device = device
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_seq_length, d_model).to(device) # pe means positional encoding
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1).to(device)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).to(device)
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        """
        Forward pass of the Positional Encoding module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_model).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, d_model).
        """
        # if x is not moved to device, move it
        if x.device.type != self.device:
            x = x.to(self.device)
        return x + self.pe[:, :x.size(1)]
    
class EncoderLayer(nn.Module):
    """
    Encoder Layer module.
    
    Args:
        d_model (int): Hidden dimension of the input tensor.
        num_heads (int): Number of attention heads.
        d_ff (int): Hidden dimension of the output tensor.
        dropout (float): Dropout probability.
        device (str, optional): Device to use.
        
    1. Calculate the masked self-attention output and add it to the input tensor, followed by dropout and layer normalization.
    2. Compute the cross-attention output between the decoder and encoder outputs, and add it to the normalized masked self-attention output, followed by dropout and layer normalization.
    3. Calculate the position-wise feed-forward output and combine it with the normalized cross-attention output, followed by dropout and layer normalization.
    4. Return the processed tensor.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float, device: str=device):
        super(EncoderLayer, self).__init__()
        self.device = device
        self.self_attn = MultiHeadAttention(d_model, num_heads, device)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, device)
        self.norm1 = nn.LayerNorm(d_model).to(device)
        self.norm2 = nn.LayerNorm(d_model).to(device)
        self.dropout = nn.Dropout(dropout).to(device)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        """
        Forward pass of the Encoder Layer module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_model).
            mask (torch.Tensor, optional): Mask tensor of shape (batch_size, seq_length, seq_length). Defaults to None.
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, d_model).
        """
        # if x is not moved to device, move it
        if x.device.type != self.device:
            x = x.to(self.device)
            
        # apply self-attention and add residual connection
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float, device: str=device):
        """
        Decoder Layer module.
        
        Args:
            d_model (int): Hidden dimension of the input tensor.
            num_heads (int): Number of attention heads.
            d_ff (int): Hidden dimension of the output tensor.
            dropout (float): Dropout probability.
            device (str, optional): Device to use.
        """
        super(DecoderLayer, self).__init__()
        self.device = device
        self.self_attn = MultiHeadAttention(d_model, num_heads, device)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, device)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, device)
        self.norm1 = nn.LayerNorm(d_model).to(device)
        self.norm2 = nn.LayerNorm(d_model).to(device)
        self.norm3 = nn.LayerNorm(d_model).to(device)
        self.dropout = nn.Dropout(dropout).to(device)
        
    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, src_mask: torch.Tensor=None, tgt_mask: torch.Tensor=None) -> torch.Tensor:
        """
        Forward pass of the Decoder Layer module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_model).
            enc_output (torch.Tensor): Encoder output tensor of shape (batch_size, seq_length, d_model).
            src_mask (torch.Tensor, optional): Source mask tensor of shape (batch_size, seq_length, seq_length). Defaults to None.
            tgt_mask (torch.Tensor, optional): Target mask tensor of shape (batch_size, seq_length, seq_length). Defaults to None.
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, d_model).
        """
        # if x is not moved to device, move it
        if x.device.type != self.device:
            x = x.to(self.device)
            
        # apply self-attention and add residual connection
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # apply cross-attention and add residual connection
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class Transformer(nn.Module):
    '''
    Transformer module.
    
    Args:
        src_vocab_size (int): Source vocabulary size.
        tgt_vocab_size (int): Target vocabulary size.
        d_model (int): Hidden dimension of the input tensor.
        num_heads (int): Number of attention heads.
        num_layers (int): Number of encoder/decoder layers.
        d_ff (int): Hidden dimension of the output tensor.
        max_seq_length (int): Maximum sequence length.
        dropout (float): Dropout probability.
        device (str, optional): Device to use.
        
    Combine the encoder and decoder modules to create the Transformer model.
    '''
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, device: str=device):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model).to(device)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model).to(device)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, device)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout, device) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout, device) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size).to(device)
        self.dropout = nn.Dropout(dropout).to(device)
        self.device = device

    def generate_mask(self, src : torch.Tensor, tgt : torch.Tensor):
        '''
        Generate source and target masks.
        
        Args:
            src (torch.Tensor): Source tensor of shape (batch_size, seq_length).
            tgt (torch.Tensor): Target tensor of shape (batch_size, seq_length).
            
        Returns:
            src_mask (torch.Tensor): Source mask tensor of shape (batch_size, 1, 1, seq_length).
            tgt_mask (torch.Tensor): Target mask tensor of shape (batch_size, 1, seq_length, seq_length).
            
        Note:
            生成源和目标Mask.
        '''
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(self.device)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3).to(self.device)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool().to(self.device)
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src : torch.Tensor, tgt : torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of the Transformer model.
        
        Args:
            src (torch.Tensor): Source tensor of shape (batch_size, seq_length).
            tgt (torch.Tensor): Target tensor of shape (batch_size, seq_length).
            
        Returns:
            output (torch.Tensor): Output tensor of shape (batch_size, seq_length, tgt_vocab_size).
            
        Note:
            Transformer模型的前向传播.
            src_embedded是经过嵌入层和位置编码层的源序列.
            tgt_embedded是经过嵌入层和位置编码层的目标序列.
            
            enc_output是经过编码器的输出.
            dec_output是经过解码器的输出.
            
            output是经过全连接层的输出.
        '''
        src = src.to(self.device)
        tgt = tgt.to(self.device)
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output

if __name__ == "__main__":
    # test
    print(f"PyTorch version: {torch.__version__}")
    print(f"Using device: {device}")
