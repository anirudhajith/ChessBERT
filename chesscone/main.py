import sys
import pinecone
import numpy as np
import chess
from utils import board_to_bitboard, encode_bitboard
import random
from multiprocessing import Pool, Manager


pinecone.init(api_key="38132697-8f87-4930-a355-376bd93394a3", environment="us-east4-gcp")
index = pinecone.Index("chesspos-lichess-embeddings")

def play_game(input):
    queue, game_count = input
    for _ in range(game_count):
        print("Starting game")
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
                    queue.put("tie")
                else:
                    print("Chesscone" if randomHasMove else "Random", "wins!", end=" ")
                    if randomHasMove:
                        queue.put("chesscone")
                    else:
                        queue.put("random")
                print("Found", foundLegalMoves, "legal moves out of", board.fullmove_number, "moves")
                break
            if randomHasMove:
                move = random.choice(list(board.legal_moves))
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

            board.push(move)
            randomHasMove = not randomHasMove

if __name__ == '__main__':
    chesscone_wins = 0
    random_wins = 0
    ties = 0

    manager = Manager()
    queue = manager.Queue()
    threading_list = []
    thread_count = 30
    game_count = 10
    with Pool(thread_count) as p:
        r = p.map_async(play_game, [[queue, game_count]] * thread_count)

        while not r.ready() or not r.successful() or queue.empty() == False:
            try:
                result = queue.get(timeout=10)
            except:
                continue
            # print to stderror so that it doesn't interfere with the output
            print(".", file=sys.stderr)
            if result == "chesscone":
                chesscone_wins += 1
            elif result == "random":
                random_wins += 1
            else:
                ties += 1
        
        r.wait()


    # Print statistics
    print("Chesscone wins:", chesscone_wins)
    print("Random wins:", random_wins)
    print("Ties:", ties)
    print("Chesscone win rate:", chesscone_wins / (chesscone_wins + random_wins))
    print("Random win rate:", random_wins / (chesscone_wins + random_wins))
    print("Tie rate:", ties / (chesscone_wins + random_wins + ties))
