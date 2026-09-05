import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import VOCAB_SIZE


model_config = {
    "num_layers": 8,
    "num_heads": 8,
    "num_kv_heads": 4,
    "hidden_dim": 768,
    "max_seq_len": 512,
    "vocab_size": VOCAB_SIZE,
    "dropout": 0.1,
}

# Rotary positional encoding
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)
 
def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)
 
class RotaryPositionalEncoding(nn.Module):
    def __init__(self, dim, max_seq_len=1024):
        super().__init__()
        N = 10000
        inv_freq = 1. / (N ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(max_seq_len).float()
        inv_freq = torch.cat((inv_freq, inv_freq), dim=-1)
        sinusoid_inp = torch.outer(position, inv_freq)
        self.register_buffer("cos", sinusoid_inp.cos())
        self.register_buffer("sin", sinusoid_inp.sin())
 
    def forward(self, x, seq_len=None):
        # x is (batch, num_heads, seq_len, head_dim): rotate along the sequence axis
        if seq_len is None:
            seq_len = x.size(-2)
        cos = self.cos[:seq_len].view(1, 1, seq_len, -1)
        sin = self.sin[:seq_len].view(1, 1, seq_len, -1)
        return apply_rotary_pos_emb(x, cos, sin)


# Neural network components
class SwiGLU(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, intermediate_dim)
        self.up = nn.Linear(hidden_dim, intermediate_dim)
        self.down = nn.Linear(intermediate_dim, hidden_dim)
        self.act = nn.SiLU()
 
    def forward(self, x):
        x = self.act(self.gate(x)) * self.up(x)
        x = self.down(x)
        return x


class GQA(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_kv_heads=None, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_groups = num_heads // self.num_kv_heads
        self.dropout = dropout
        # K and V carry only num_kv_heads heads, each shared by num_groups query heads
        kv_dim = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, kv_dim)
        self.v_proj = nn.Linear(hidden_dim, kv_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
 
    def forward(self, q, k, v, mask=None, rope=None):
        q_batch_size, q_seq_len, hidden_dim = q.shape
        k_batch_size, k_seq_len, hidden_dim = k.shape
        v_batch_size, v_seq_len, hidden_dim = v.shape
 
        # projection
        q = self.q_proj(q).view(q_batch_size, q_seq_len, -1, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(k_batch_size, k_seq_len, -1, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(v_batch_size, v_seq_len, -1, self.head_dim).transpose(1, 2)
 
        # apply rotary positional encoding
        if rope:
            q = rope(q)
            k = rope(k)
 
        # compute grouped query attention
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        output = F.scaled_dot_product_attention(q, k, v,
                                                attn_mask=mask,
                                                dropout_p=self.dropout if self.training else 0.0,
                                                enable_gqa=True)
        output = output.transpose(1, 2).reshape(q_batch_size, q_seq_len, hidden_dim).contiguous()
        output = self.out_proj(output)
        return output


class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_kv_heads, dropout=0.1):
        super().__init__()
        self.self_attn = GQA(hidden_dim, num_heads, num_kv_heads, dropout)
        self.mlp = SwiGLU(hidden_dim, 4 * hidden_dim)
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)
 
    def forward(self, x, mask=None, rope=None):
        # self-attention sublayer
        out = self.norm1(x)
        out = self.self_attn(out, out, out, mask, rope)
        x = out + x
        # MLP sublayer
        out = self.norm2(x)
        out = self.mlp(out)
        return out + x


class TextGenerationModel(nn.Module):
    def __init__(self, num_layers, num_heads, num_kv_heads, hidden_dim,
                 max_seq_len, vocab_size, dropout=0.1):
        super().__init__()
        self.rope = RotaryPositionalEncoding(hidden_dim // num_heads, max_seq_len)
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.decoders = nn.ModuleList([
            DecoderLayer(hidden_dim, num_heads, num_kv_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.RMSNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)
 
    def forward(self, ids, mask=None):
        x = self.embedding(ids)
        for decoder in self.decoders:
            x = decoder(x, mask, self.rope)
        x = self.norm(x)
        return self.out(x)

def create_model(device=None) -> nn.Module:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.compile(TextGenerationModel(**model_config)).to(device)
    return model

def create_causal_mask(seq_len, device=None):
    """Create a causal mask for autoregressive attention."""
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)
    return mask


if __name__ == "__main__":
    # Exercise create_model
    model = create_model('cpu')
