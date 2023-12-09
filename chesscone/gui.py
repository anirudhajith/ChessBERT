import pinecone
import numpy as np
import chess
from utils import board_to_bitboard, encode_bitboard
import random
import chess.svg

pinecone.init(api_key="REDACTED", environment="us-east4-gcp")
index = pinecone.Index("chesspos-lichess-embeddings")

chesscone_wins = 0
random_wins = 0
ties = 0

for i in range(100):
    randomHasMove = random.random() < 0.5  # Is it the random player's turn?
    foundLegalMoves = 0
    board = chess.Board()
    while True:
        # Check if game is over, or if there is a draw, or if there is a 100 move rule
        if board.is_game_over() or board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves():
            # Print name of the winner, random if the random player won, otherwise chesscone
            # If tie, print "Tie"
            if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves():
                print("Tie!", end=" ")
                ties += 1
            else:
                print("Chesscone" if randomHasMove else "Random", "wins!", end=" ")
                if randomHasMove:
                    chesscone_wins += 1
                else:
                    random_wins += 1
            print("Found", foundLegalMoves, "legal moves out of", board.fullmove_number, "moves")
            break
        if randomHasMove:
            svg = chess.svg.board(board=board, size=600)
            with open("board.svg", "w") as f:
                f.write(svg)
            move = input("Enter move: ")
            move = chess.Move.from_uci(move)
        else:
            bitboard = board_to_bitboard(board)
            embedding = encode_bitboard(np.array([bitboard]))
            results = index.query(vector=embedding.tolist(), top_k=30, include_metadata=True)
            for result in results["matches"]:
                move = chess.Move.from_uci(result["metadata"]["move"])
                if move in board.legal_moves:
                    foundLegalMoves += 1
                    break
            else:
                move = random.choice(list(board.legal_moves))
                print("Made random move:", move.uci())

        board.push(move)
        randomHasMove = not randomHasMove
        
# Print statistics
print("Chesscone wins:", chesscone_wins)
print("Random wins:", random_wins)
print("Ties:", ties)
print("Chesscone win rate:", chesscone_wins / (chesscone_wins + random_wins))
print("Random win rate:", random_wins / (chesscone_wins + random_wins))
print("Tie rate:", ties / (chesscone_wins + random_wins + ties))
