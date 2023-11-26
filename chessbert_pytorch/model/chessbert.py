import torch.nn as nn

from .transformer import TransformerBlock
from .embedding import PieceSquareEmbedding


class ChessBERT(nn.Module):
    """
    ChessBERT model : BERT-based retrieval-based chess engine
    """

    def __init__(self, hidden=128, n_layers=12, attn_heads=12, dropout=0.1):
        """
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = PieceSquareEmbedding(embed_size=hidden)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x):
        """
        :param x: (batch_size, seq_len, 4)
        """

        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        mask = (x[:,:,0] > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x) # (batch_size, seq_len, hidden)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask) # (batch_size, seq_len, hidden)

        return x
