import torch
import torch.nn as nn

VOCAB = 64
DIM = 32
LAYERS = 1
HEADS = 4


class TinyTransformer(nn.Module):
    """Small transformer used as the student model. Kept tiny on purpose so
    a full reference + resumed-run comparison runs in seconds on CPU."""

    def __init__(self, seed: int = 0):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, DIM)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=DIM, nhead=HEADS, dim_feedforward=64, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=LAYERS)
        self.head = nn.Linear(DIM, VOCAB)
        self._deterministic_init(seed)

    def _deterministic_init(self, seed: int):
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p, generator=g)
                else:
                    nn.init.zeros_(p)

    def forward(self, x):
        h = self.embed(x)
        h = self.encoder(h)
        return self.head(h)
