import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.r_pos = nn.ParameterDict({
            "relative": nn.Parameter(pe, requires_grad=False)
        })

    def forward(self, x):
        pe = self.r_pos["relative"]
        x_len = x.shape[1]
        return x + pe[:, :x_len, :]


class LowRankMultiheadAttention(nn.Module):
    def __init__(self, in_dim, embed_dim,
                 num_heads, dropout=0.0, max_len=1501, q_in_dim=None):
        super(LowRankMultiheadAttention, self).__init__()

        self.dropout = dropout

        if q_in_dim is None:
            q_in_dim = in_dim

        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads

        self.q_linear = nn.Linear(q_in_dim, embed_dim, bias=True)
        self.k_linear = nn.Linear(in_dim, embed_dim, bias=True)
        self.v_linear = nn.Linear(in_dim, embed_dim, bias=True)

        self.pos_linear = nn.Linear(in_dim, embed_dim, bias=True)

        self.pos = PositionalEncoding(d_model=in_dim, max_len=max_len)


        self.gates = nn.Parameter(torch.zeros([1]), requires_grad=True)
        self.attn_dropout = nn.Dropout(dropout)

        self.out_proj = nn.Linear(embed_dim, q_in_dim, bias=True)
    def forward(self, query, key, value, attn_mask, indices_query, indices_key):


        num_heads = self.num_heads
        head_dim = self.head_dim

        batch_size = len(key)
        key_len = key.shape[1]
        query_len = query.shape[1]
        # print('Key len:', key_len, 'Query len:', query_len)
        pos_query = self.pos_linear(self.pos.r_pos["relative"][:, indices_query])
        pos_key = self.pos_linear(self.pos.r_pos["relative"][:, indices_key])
        key = (self.k_linear(key) + pos_key).view(-1, key_len, num_heads, head_dim).transpose(1, 2)
        value = (self.v_linear(value) + pos_key).view(-1, key_len, num_heads, head_dim).transpose(1, 2)
        query = (self.q_linear(query) + pos_query).view(-1, query_len, num_heads, head_dim).transpose(1, 2)

        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / (head_dim ** 0.5)
        if attn_mask is not None:
            if attn_mask.dim() == 2:  # If shape is (t_q, t_k), broadcast to (batch_size, num_heads, t_q, t_k)
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # Shape becomes (1, 1, t_q, t_k)
            attn_weights = attn_weights + attn_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        return self.out_proj(attn_output) * self.gates
