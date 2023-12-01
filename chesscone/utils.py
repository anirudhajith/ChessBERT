import numpy as np
import chess
import tensorflow as tf

model = None

def encode_bitboard(query):
    global model
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
