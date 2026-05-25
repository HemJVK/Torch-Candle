import copy
from ..module import Module
from ..linear import Linear
from ..activations import ReLU
from .normalization import LayerNorm
from ..dropout import Dropout
from .. import functional as F

class MultiheadAttention(Module):
    """
    Multi-Head Attention layer.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True,
                 add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None,
                 batch_first=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.batch_first = batch_first

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None, is_causal=False):
        # query, key, value shape: [seq_len, batch, embed_dim] (if batch_first=False)
        # or [batch, seq_len, embed_dim] (if batch_first=True)
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        seq_len, batch_size, embed_dim = query.shape
        k_seq_len = key.shape[0]

        # Project query, key, value
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape & Transpose query, key, value natively on the GPU/CPU with zero PCI-E copies!
        # [seq_len, batch_size, embed_dim] -> [seq_len, batch_size, num_heads, head_dim]
        # -> transpose(0, 1) -> [batch_size, seq_len, num_heads, head_dim]
        # -> transpose(1, 2) -> [batch_size, num_heads, seq_len, head_dim]
        q_t = q.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1).transpose(1, 2)
        k_t = k.view(k_seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1).transpose(1, 2)
        v_t = v.view(k_seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1).transpose(1, 2)

        # Scale dot product attention
        attn_out = F.scaled_dot_product_attention(
            q_t, k_t, v_t, attn_mask=attn_mask, dropout_p=self.dropout, is_causal=is_causal
        )

        # Reshape back to [seq_len, batch, embed_dim] 100% natively!
        # attn_out shape is [batch, heads, seq, head_dim]
        # -> transpose(1, 2) -> [batch, seq, heads, head_dim]
        # -> transpose(0, 1) -> [seq, batch, heads, head_dim]
        # -> view -> [seq_len, batch_size, embed_dim]
        out = attn_out.transpose(1, 2).transpose(0, 1).view(seq_len, batch_size, embed_dim)

        # Final linear projection
        out = self.out_proj(out)

        if self.batch_first:
            out = out.transpose(0, 1)

        # Return a tuple matching PyTorch signature: (attn_output, attn_output_weights)
        return out, None


class TransformerEncoderLayer(Module):
    """
    Transformer Encoder Layer.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation=ReLU, layer_norm_eps=1e-5, norm_first=False):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)
        
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        
        self.activation = activation() if isinstance(activation, type) else activation
        self.norm_first = norm_first

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        x = src
        if self.norm_first:
            # Pre-LN
            x = x + self.dropout1(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x),
                                                 attn_mask=src_mask, is_causal=is_causal)[0])
            x = x + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(self.norm2(x))))))
        else:
            # Post-LN
            x = self.norm1(x + self.dropout1(self.self_attn(x, x, x, attn_mask=src_mask, is_causal=is_causal)[0]))
            x = self.norm2(x + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x))))))
        return x


class TransformerEncoder(Module):
    """
    Transformer Encoder.
    """
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = []
        for i in range(num_layers):
            layer = copy.deepcopy(encoder_layer)
            self.layers.append(layer)
            self.add_module(f"layer_{i}", layer)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None, is_causal=False):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, is_causal=is_causal)
        if self.norm is not None:
            output = self.norm(output)
        return output
