# Table of Contents

* [main](#main)
  * [MultiHeadAttention](#main.MultiHeadAttention)
    * [\_\_init\_\_](#main.MultiHeadAttention.__init__)
    * [scaled\_dot\_product\_attention](#main.MultiHeadAttention.scaled_dot_product_attention)
    * [split\_heads](#main.MultiHeadAttention.split_heads)
    * [combine\_heads](#main.MultiHeadAttention.combine_heads)
    * [forward](#main.MultiHeadAttention.forward)
  * [PositionWiseFeedForward](#main.PositionWiseFeedForward)
    * [\_\_init\_\_](#main.PositionWiseFeedForward.__init__)
    * [forward](#main.PositionWiseFeedForward.forward)
  * [PositionalEncoding](#main.PositionalEncoding)
    * [\_\_init\_\_](#main.PositionalEncoding.__init__)
    * [forward](#main.PositionalEncoding.forward)
  * [EncoderLayer](#main.EncoderLayer)
    * [\_\_init\_\_](#main.EncoderLayer.__init__)
    * [forward](#main.EncoderLayer.forward)
  * [DecodeerLayer](#main.DecodeerLayer)
    * [\_\_init\_\_](#main.DecodeerLayer.__init__)
    * [forward](#main.DecodeerLayer.forward)

<a id="main"></a>

# main

<a id="main.MultiHeadAttention"></a>

## MultiHeadAttention Objects

```python
class MultiHeadAttention(nn.Module)
```

Multi-Head Attention module.

**Arguments**:

- `d_model` _int_ - Hidden dimension of the input tensor.
- `num_heads` _int_ - Number of attention heads.
- `device` _str, optional_ - Device to use.
  计算多头自注意力，使模型能够关注输入序列的某些不同方面

<a id="main.MultiHeadAttention.__init__"></a>

#### \_\_init\_\_

```python
def __init__(d_model: int, num_heads: int, device: str = device)
```

Multi-Head Attention module.

<a id="main.MultiHeadAttention.scaled_dot_product_attention"></a>

#### scaled\_dot\_product\_attention

```python
def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor
```

Scaled Dot-Product Attention mechanism.

**Arguments**:

- `Q` _torch.Tensor_ - Query tensor of shape (batch_size, num_queries, d_model).
- `K` _torch.Tensor_ - Key tensor of shape (batch_size, num_keys, d_model).
- `V` _torch.Tensor_ - Value tensor of shape (batch_size, num_values, d_model).
- `mask` _torch.Tensor, optional_ - Mask tensor of shape (batch_size, num_queries, num_keys). Defaults to None.
  

**Returns**:

- `torch.Tensor` - Output tensor of shape (batch_size, num_queries, d_model).

<a id="main.MultiHeadAttention.split_heads"></a>

#### split\_heads

```python
def split_heads(x)
```

Split the last dimension of the input tensor into (num_heads, d_k).

**Arguments**:

- `x` _torch.Tensor_ - Input tensor of shape (batch_size, seq_length, d_model).
  

**Returns**:

- `torch.Tensor` - Output tensor of shape (batch_size, num_heads, seq_length, d_k).

<a id="main.MultiHeadAttention.combine_heads"></a>

#### combine\_heads

```python
def combine_heads(x)
```

Combine the (num_heads, d_k) dimensions of the input tensor into a single dimension.

**Arguments**:

- `x` _torch.Tensor_ - Input tensor of shape (batch_size, num_heads, seq_length, d_k).
  

**Returns**:

- `torch.Tensor` - Output tensor of shape (batch_size, seq_length, d_model).

<a id="main.MultiHeadAttention.forward"></a>

#### forward

```python
def forward(Q, K, V, mask=None)
```

Forward pass of the Multi-Head Attention module.

**Arguments**:

- `Q` _torch.Tensor_ - Query tensor of shape (batch_size, num_queries, d_model).
- `K` _torch.Tensor_ - Key tensor of shape (batch_size, num_keys, d_model).
- `V` _torch.Tensor_ - Value tensor of shape (batch_size, num_values, d_model).
- `mask` _torch.Tensor, optional_ - Mask tensor of shape (batch_size, num_queries, num_keys). Defaults to None.
  

**Returns**:

- `torch.Tensor` - Output tensor of shape (batch_size, num_queries, d_model).

<a id="main.PositionWiseFeedForward"></a>

## PositionWiseFeedForward Objects

```python
class PositionWiseFeedForward(nn.Module)
```

<a id="main.PositionWiseFeedForward.__init__"></a>

#### \_\_init\_\_

```python
def __init__(d_model: int, d_ff: int, device: str = device)
```

Position-Wise Feed-Forward module.

**Arguments**:

- `d_model` _int_ - Hidden dimension of the input tensor.
- `d_ff` _int_ - Hidden dimension of the output tensor.
- `device` _str, optional_ - Device to use.
  此过程使模型能够在进行预测时考虑输入元素的位置。
  
  Transformer中的FFN全称是Position-wise Feed-Forward Networks，重点就是这个position-wise，区别于普通的全连接网络，这里FFN的输入是序列中每个位置上的元素，而不是整个序列，所以每个元素完全可以独立计算，最极端节省内存的做法是遍历序列，每次只取一个元素得到FFN的结果，但是这样做时间消耗太大，“分段”的含义就是做下折中，将序列分成段，也就是个子序列，每次读取一个子序列进行FFN计算，最后将份的结果拼接。分段FFN只是一种计算上的技巧，计算结果和原始FFN完全一致，所以不会影响到模型效果，好处是不需要一次性将整个序列读入内存，劣势当然是会增加额外的时间开销了。

<a id="main.PositionWiseFeedForward.forward"></a>

#### forward

```python
def forward(x: torch.Tensor) -> torch.Tensor
```

Forward pass of the Position-Wise Feed-Forward module.

**Arguments**:

- `x` _torch.Tensor_ - Input tensor of shape (batch_size, seq_length, d_model).
  

**Returns**:

- `torch.Tensor` - Output tensor of shape (batch_size, seq_length, d_model).

<a id="main.PositionalEncoding"></a>

## PositionalEncoding Objects

```python
class PositionalEncoding(nn.Module)
```

<a id="main.PositionalEncoding.__init__"></a>

#### \_\_init\_\_

```python
def __init__(d_model: int, max_seq_length: int, device: str = device)
```

Positional Encoding module.

$$
PE_{(pos, 2i)} = sin(pos / 10000^{2i / d_{model}})
PE_{(pos, 2i+1)} = cos(pos / 10000^{2i / d_{model}})
$$

**Arguments**:

- `d_model` _int_ - Hidden dimension of the input tensor.
- `max_seq_length` _int_ - Maximum sequence length.
- `device` _str, optional_ - Device to use.
  为输入序列中的每个位置添加一个可学习的向量，以便模型能够考虑序列中元素的顺序。

<a id="main.PositionalEncoding.forward"></a>

#### forward

```python
def forward(x)
```

Forward pass of the Positional Encoding module.

**Arguments**:

- `x` _torch.Tensor_ - Input tensor of shape (batch_size, seq_length, d_model).
  

**Returns**:

- `torch.Tensor` - Output tensor of shape (batch_size, seq_length, d_model).

<a id="main.EncoderLayer"></a>

## EncoderLayer Objects

```python
class EncoderLayer(nn.Module)
```

<a id="main.EncoderLayer.__init__"></a>

#### \_\_init\_\_

```python
def __init__(d_model: int, num_heads: int, d_ff: int, dropout: float, device: str = device)
```

Encoder Layer module.

**Arguments**:

- `d_model` _int_ - Hidden dimension of the input tensor.
- `num_heads` _int_ - Number of attention heads.
- `d_ff` _int_ - Hidden dimension of the output tensor.
- `dropout` _float_ - Dropout probability.
- `device` _str, optional_ - Device to use.

<a id="main.EncoderLayer.forward"></a>

#### forward

```python
def forward(x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor
```

Forward pass of the Encoder Layer module.

**Arguments**:

- `x` _torch.Tensor_ - Input tensor of shape (batch_size, seq_length, d_model).
- `mask` _torch.Tensor, optional_ - Mask tensor of shape (batch_size, seq_length, seq_length). Defaults to None.
  

**Returns**:

- `torch.Tensor` - Output tensor of shape (batch_size, seq_length, d_model).

<a id="main.DecodeerLayer"></a>

## DecodeerLayer Objects

```python
class DecodeerLayer(nn.Module)
```

<a id="main.DecodeerLayer.__init__"></a>

#### \_\_init\_\_

```python
def __init__(d_model: int, num_heads: int, d_ff: int, dropout: float, device: str = device)
```

Decoder Layer module.

**Arguments**:

- `d_model` _int_ - Hidden dimension of the input tensor.
- `num_heads` _int_ - Number of attention heads.
- `d_ff` _int_ - Hidden dimension of the output tensor.
- `dropout` _float_ - Dropout probability.
- `device` _str, optional_ - Device to use.

<a id="main.DecodeerLayer.forward"></a>

#### forward

```python
def forward(x: torch.Tensor, enc_output: torch.Tensor, src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None) -> torch.Tensor
```

Forward pass of the Decoder Layer module.

**Arguments**:

- `x` _torch.Tensor_ - Input tensor of shape (batch_size, seq_length, d_model).
- `enc_output` _torch.Tensor_ - Encoder output tensor of shape (batch_size, seq_length, d_model).
- `src_mask` _torch.Tensor, optional_ - Source mask tensor of shape (batch_size, seq_length, seq_length). Defaults to None.
- `tgt_mask` _torch.Tensor, optional_ - Target mask tensor of shape (batch_size, seq_length, seq_length). Defaults to None.
  

**Returns**:

- `torch.Tensor` - Output tensor of shape (batch_size, seq_length, d_model).

