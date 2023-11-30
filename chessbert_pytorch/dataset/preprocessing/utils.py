import numpy as np
import h5py
import pickle
import chess

def board_to_array(board):
    board_arr = np.zeros(64)
    for color in [0, 1]:
        for i in range(1,7): #P N B R Q K
            for j in list(board.pieces(i, color)):
                if color == 0:
                    board_arr[j] = i * -1
                else:
                    board_arr[j] = i

    #print(board.fen())
    #print(board_arr.reshape((8,8)))

    additional = np.array([
        bool(board.turn),
        bool(board.castling_rights & chess.BB_A1),
        bool(board.castling_rights & chess.BB_H1),
        bool(board.castling_rights & chess.BB_A8),
        bool(board.castling_rights & chess.BB_H8)
    ]).astype(np.float64)

    board_arr = np.concatenate((board_arr, additional))
    return board_arr

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
