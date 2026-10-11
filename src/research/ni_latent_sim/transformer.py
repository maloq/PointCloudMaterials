"""Small causal transformer for the four observed/predicted latent tokens."""
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, width, heads, feedforward):
        super().__init__()
        if width % heads:
            raise ValueError('Attention width must divide evenly across heads')
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3*width)
        self.output = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(nn.Linear(width, feedforward), nn.GELU(), nn.Linear(feedforward, width))

    def forward(self, tokens):
        batch, length, width = tokens.shape
        qkv = self.qkv(self.norm1(tokens)).reshape(batch, length, 3, self.heads, width//self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0., is_causal=True)
        tokens = tokens + self.output(attended.transpose(1, 2).reshape(batch, length, width))
        return tokens + self.feedforward(self.norm2(tokens))


class TemporalTransformer(nn.Module):
    def __init__(self, dimension, memory, settings):
        super().__init__()
        self.dimension, self.memory = dimension, memory
        width = settings['width']
        self.projection = nn.Linear(dimension, width)
        # Ordered slots replace the ordering already implicit in MLP concatenation.
        # No timestamps, physical ages, temperatures or substep phases are inputs.
        self.slots = nn.Parameter(torch.empty(memory, width))
        nn.init.normal_(self.slots, std=.02)
        self.blocks = nn.ModuleList([Block(width, settings['heads'], settings['feedforward'])
                                     for _ in range(settings['layers'])])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 2*dimension)
        nn.init.normal_(self.head.weight, std=.001)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[dimension:] = -3

    def forward(self, history):
        leading = history.shape[:-1]
        tokens = self.projection(history.reshape(-1, self.memory, self.dimension)) + self.slots
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(self.norm(tokens[:, -1])).reshape(*leading, 2*self.dimension)
