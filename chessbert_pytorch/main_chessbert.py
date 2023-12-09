import sys
import pinecone
import numpy as np
import chess
import random
from multiprocessing import Pool, Manager
import tensorflow as tf
from get_prediction import get_prediction




def encode_bitboard(query):
    if model is None:
        model = tf.keras.models.load_model("model_encoder.h5")
    embedding = model.predict_on_batch(query)
    return np.asarray(embedding, dtype=np.float32)

def board_to_bitboard(board):
	embedding = np.array([], dtype=bool)
	for color in [1, 0]:
		for i in range(1, 7): # P N B R Q K / white
			bmp = np.zeros(shape=(64,)).astype(bool)
			for j in list(board.pieces(i, color)):
				bmp[j] = True
			embedding = np.concatenate((embedding, bmp))
	additional = np.array([
		bool(board.turn),
		bool(board.castling_rights & chess.BB_A1),
		bool(board.castling_rights & chess.BB_H1),
		bool(board.castling_rights & chess.BB_A8),
		bool(board.castling_rights & chess.BB_H8)
	])
	embedding = np.concatenate((embedding, additional))
	return embedding


def play_game(input):
    index = pinecone.Index("chesspos-lichess-embeddings")
    model = tf.keras.models.load_model("dataset/preprocessing/model_encoder.h5")

    queue, game_count = input
    tie_count = 0
    w_count = 0
    l_count = 0
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
                    tie_count += 1
                else:
                    print("Chesscone" if randomHasMove else "Random", "wins!", end=" ")
                    if randomHasMove:
                        queue.put("chesscone")
                        w_count += 1
                    else:
                        queue.put("random")
                        l_count += 1
                print("Found", foundLegalMoves, "legal moves out of", board.fullmove_number, "moves")
                break
            if randomHasMove:
                move = random.choice(list(board.legal_moves))
            else:
                legal_moves = [move for move in board.legal_moves]
                move_index, _ = get_prediction(board.fen(), [move.uci() for move in legal_moves], model, index)
                move = legal_moves[move_index]

            board.push(move)
            randomHasMove = not randomHasMove
        print("%d - %d - %d\n" % (w_count, l_count, tie_count))

if __name__ == '__main__':
    chesscone_wins = 0
    random_wins = 0
    ties = 0

    manager = Manager()
    queue = manager.Queue()
    threading_list = []
    thread_count = 30
    game_count = 100
    play_game([queue, game_count])
    while queue.empty() == False:
        result = queue.get()
        # print to stderror so that it doesn't interfere with the output
        if result == "chesscone":
            chesscone_wins += 1
        elif result == "random":
            random_wins += 1
        else:
            ties += 1

    # Print statistics
    print("Chesscone wins:", chesscone_wins)
    print("Random wins:", random_wins)
    print("Ties:", ties)
    print("Chesscone win rate:", chesscone_wins / (chesscone_wins + random_wins))
    print("Random win rate:", random_wins / (chesscone_wins + random_wins))
    print("Tie rate:", ties / (chesscone_wins + random_wins + ties))
