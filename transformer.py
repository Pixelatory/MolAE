import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0, bias=True, 
                 kdim=None, vdim=None, scale=None) -> None:
        super().__init__()
        kdim = kdim if kdim is not None else d_model
        vdim = vdim if vdim is not None else d_model
        self.scale = scale if scale is not None else d_model ** 0.5
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model
        assert self.head_dim * self.n_heads == self.d_model, "d_model must be divisible by num_heads"

        self.query_proj = nn.Linear(d_model, d_model, bias=bias)
        self.key_proj = nn.Linear(kdim, d_model, bias=bias)
        self.value_proj = nn.Linear(vdim, d_model, bias=bias)

        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor=None):
        """
        Parameters:
        - mask: Masks out attention to values where mask is set to 1. Should be shape (query_len x key_len/value_len), \
            (batch_size x n_heads x query_len x key_len/value_len), or (batch_size x key_len/value_len)
        """
        batch_size = query.size(0)
        Q = self.query_proj(query)  # batch x seq_len x query_len
        K = self.key_proj(key)  # batch x seq_len x key_len
        V = self.value_proj(value)  # batch x seq_len x value_len

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # batch_size x n_heads x query_len x head_dim
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # batch_size x n_heads x key_len x head_dim
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # batch_size x n_heads x value_len x head_dim

        energy = (Q @ K.permute(0, 1, 3, 2)) / self.scale  # batch_size x n_heads x query_len x (key_len/value_len)
        if mask is not None:
            print('mask', mask)
            if list(mask.shape) == [energy.shape[0], energy.shape[-1]]:
                mask = mask.unsqueeze(1).unsqueeze(1)  # source sequence padding mask: batch_size x 1 x 1 x (key_len/value_len)
            energy = energy.masked_fill_(mask == 1, float('-inf'))
        
        attention = energy.softmax(-1)
        x = self.dropout(attention) @ V  # batch_size x n_heads x query_len x head_dim
        x = x.permute(0, 2, 1, 3).contiguous()  # batch_size x query_len x n_heads x head_dim
        x = x.view(batch_size, -1, self.d_model)  # batch_size x query_len x d_model
        return self.out_proj(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int,
                 dim_feedforward: int, dropout: float,
                 pre_norm: bool = False,
                 activation_fn: nn.Module = nn.ReLU,
                 norm_fn: nn.Module = nn.LayerNorm,
                 bias: bool = True, **kwargs) -> None:
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout, bias=bias, **kwargs)
        self.norm1 = norm_fn(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward, bias=bias),
            activation_fn(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model, bias=bias)
        )
        self.dropout = nn.Dropout(dropout)
        self.norm2 = norm_fn(d_model)
        self.pre_norm = pre_norm

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Parameters:
        - mask: Masks out attention to values where mask is set to 1. Should be shape (query_len x key_len/value_len), \
            (batch_size x n_heads x query_len x key_len/value_len), or (batch_size x key_len/value_len)
        """
        residual = x
        x = self.norm1(x) if self.pre_norm else x
        x = residual + self.dropout(self.attention(x, x, x, mask))
        x = self.norm1(x) if not self.pre_norm else x

        residual = x
        x = self.norm2(x) if self.pre_norm else x
        x = residual + self.dropout(self.mlp(x))
        x = self.norm2(x) if not self.pre_norm else x
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int,
                 dim_feedforward: int, dropout: float,
                 pre_norm: bool = False,
                 activation_fn: nn.Module = nn.ReLU,
                 norm_fn: nn.Module = nn.LayerNorm,
                 bias: bool = True, **kwargs) -> None:
        super().__init__()
        
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout, bias=bias, **kwargs)
        self.norm3 = norm_fn(d_model)
        self.mlp2 = nn.Sequential(
            nn.Linear(d_model, dim_feedforward, bias=bias),
            activation_fn(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model, bias=bias)
        )
        self.norm4 = norm_fn(d_model)

        self.attention = MultiHeadAttention(d_model, n_heads, dropout, bias=bias)
        self.norm1 = norm_fn(d_model)
        self.mlp1 = nn.Sequential(
            nn.Linear(d_model, dim_feedforward, bias=bias),
            activation_fn(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model, bias=bias)
        )
        self.norm2 = norm_fn(d_model)

        self.dropout = nn.Dropout(dropout)
        self.pre_norm = pre_norm

    def forward(self, encoder_out: torch.Tensor, tgt: torch.Tensor, 
                self_attn_mask: torch.Tensor = None, cross_attn_mask: torch.Tensor = None):
        """
        Parameters:
        - self_attn_mask: This is usually set to the causal mask. Masks out attention to values where mask is set to 1. \
            Should be shape (query_len x key_len/value_len), (batch_size x n_heads x query_len x key_len/value_len), \
            or (batch_size x key_len/value_len)
        - cross_attn_mask: Masks out attention to values where mask is set to 1. \
            Should be shape (query_len x key_len/value_len), (batch_size x n_heads x query_len x key_len/value_len), \
            or (batch_size x key_len/value_len)
        """
        # self-attention followed by feedforward on tgt tensor
        residual = tgt
        tgt = self.norm1(tgt) if self.pre_norm else tgt
        tgt = residual + self.dropout(self.attention(tgt, tgt, tgt, mask=self_attn_mask))
        tgt = self.norm1(tgt) if not self.pre_norm else tgt

        residual = tgt
        tgt = self.norm2(tgt) if self.pre_norm else tgt
        tgt = residual + self.dropout(self.mlp1(tgt))
        tgt = self.norm2(tgt) if not self.pre_norm else tgt

        # cross-attention followed by feedforward with encoder_out and tgt tensors
        residual = tgt
        tgt = self.norm3(tgt) if self.pre_norm else tgt
        tgt = residual + self.dropout(self.cross_attention(tgt, encoder_out, encoder_out, mask=cross_attn_mask))
        tgt = self.norm3(tgt) if not self.pre_norm else tgt

        residual = tgt
        tgt = self.norm4(tgt) if self.pre_norm else tgt
        tgt = residual + self.dropout(self.mlp2(tgt))
        tgt = self.norm4(tgt) if not self.pre_norm else tgt

        return tgt