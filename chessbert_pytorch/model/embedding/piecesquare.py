import torch.nn as nn
from .token import TokenEmbedding
from .segment import SegmentEmbedding


class PieceSquareEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. PieceIDEmbedding: embedding matrix of size (32+3)x128 representing [PAD], white a pawn, white b pawn, white c pawn, ..., white, queenside rook, white queenside knight, etc., [SEP], [MASK]
        2. RowEmbedding: adding positional information about rows: [null], 1, 2, 3, ..., 8
        3. FileEmbedding: adding position information about [null], a, b, c, ..., h
        4. SegmentEmbedding : adding board segment info about [null], [demo board 1], [next move 1], [demo board 2], [next move 2], ..., [demo board 8], [next move 8], [query board], [query next move]

        sum of all these features are output of PieceSquareEmbedding
    """

    def __init__(self, embed_size, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.piece = TokenEmbedding(1+16+16+2, embed_size)
        self.row = nn.Embedding(1+8, embed_size, padding_idx=0)
        self.file = nn.Embedding(1+8, embed_size, padding_idx=0)
        self.segment = SegmentEmbedding(self.piece.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, x):
        """
        :param x: (batch_size, seq_len, 4)
        """
        pieces, rows, files, segments = x[:,:,0].int(), x[:,:,1].int(), x[:,:,2].int(), x[:,:,3].int()
        x = self.piece(pieces) + self.row(rows) + self.file(files) + self.segment(segments) # (batch_size, seq_len, hidden)
        return self.dropout(x)
