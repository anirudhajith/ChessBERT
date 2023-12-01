import pinecone
import numpy as np
import chess
from utils import board_to_bitboard, encode_bitboard
import random
import chess.svg

pinecone.init(api_key="38132697-8f87-4930-a355-376bd93394a3", environment="us-east4-gcp")
index = pinecone.Index("chesspos-lichess-embeddings")

board_str = "r1bqk2r/ppp2ppp/1bnp1n2/4p3/2B1P3/2NPBN2/PPP2PPP/R2QK2R w KQkq - 2 7"

board = chess.Board(board_str)

bitboard = board_to_bitboard(board)
embedding = encode_bitboard(np.array([bitboard]))
results = index.query(vector=embedding.tolist(), top_k=30, include_metadata=True)
print(results)