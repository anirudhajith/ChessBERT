import torch
import torch.nn as nn

from .chessbert import ChessBERT

class MaskedChessModel(nn.Module):

    def __init__(self, hidden=128, n_layers=12, attn_heads=12):
        """
        :param hidden: BERT model hidden representation size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        """
        super().__init__()
        self.chessbert = ChessBERT(hidden, n_layers, attn_heads)
        self.piece_embeddings = self.chessbert.embedding.piece # (num_embeddings, hidden)
        self.row_embeddings = self.chessbert.embedding.row # (num_embeddings, hidden)
        self.file_embeddings = self.chessbert.embedding.file # (num_embeddings, hidden)
        self.segment_embeddings = self.chessbert.embedding.segment # (num_embeddings, hidden)

    def forward(self, x):
        """
        :param x: (batch_size, seq_len, 4)
        """
        return self.chessbert(x)[:,-1,:] # (batch_size, hidden)
    
    def predict(self, x):
        """
        :param x: (batch_size, seq_len, 4)
        """
        move_embedding = self.chessbert(x)[:,-1,:] - self.segment_embeddings.weight[-1,:] # (batch_size, hidden)
        piece_id = torch.argmax(move_embedding @ self.piece_embeddings.weight.T, dim=1) # (batch_size)
        row_id = torch.argmax(move_embedding @ self.row_embeddings.weight.T, dim=1) # (batch_size)
        file_id = torch.argmax(move_embedding @ self.file_embeddings.weight.T, dim=1) # (batch_size)
        
        return piece_id, row_id, file_id
