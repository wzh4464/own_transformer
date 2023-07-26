# Multi-Head Attention Module

The `MultiHeadAttention` module takes in query, key, and value tensors and returns an output tensor.

## Initialization

```python
def __init__(self, d_model, num_heads):
    """
    Initialize the Multi-Head Attention module.

    Args:
        d_model (int): Dimensionality of the model.
        num_heads (int): Number of attention heads.
    """
```

Initializes the `MultiHeadAttention` module with the given `d_model` and `num_heads`.

## Split Heads

```python
def split_heads(self, x):
    """
    Split the last dimension of the input tensor into (num_heads, d_k).

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_model).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, num_heads, seq_length, d_k).
    """
```

Splits the last dimension of the input tensor into `(num_heads, d_k)`.

## Combine Heads

```python
def combine_heads(self, x):
    """
    Combine the (num_heads, d_k) dimensions of the input tensor into a single dimension.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_heads, seq_length, d_k).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, seq_length, d_model).
    """
```

Combines the `(num_heads, d_k)` dimensions of the input tensor into a single dimension.

## Forward Pass

```python
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
```

Performs the forward pass of the `MultiHeadAttention` module.
