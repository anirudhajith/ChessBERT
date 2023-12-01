from .preprocessing.utils import *
import numpy as np
import h5py
import pickle
import chess
from collections import defaultdict
import pinecone
import random
from torch.utils.data import default_collate

import chess

def collate_fn(batch):
    max_len = -1
    for i in range(len(batch)):
        max_len = max(max_len, len(batch[i][0]))

    padded = []
    for i in range(len(batch)):
        x = batch[i][0]
        x = np.concatenate([np.zeros((max_len - len(x), 4)), x], axis = 0)
        padded.append((x, batch[i][1], batch[i][2]))

    return default_collate(padded)

def isBlack(r, f):
    return (r+1) % 2 == f % 2

def array_to_y(array, piece_index, segment_id, uci_moves):
    board = array[:64]
    additional = array[64:69]
    move = array[-4:]
    board = board.reshape((8,8))

    uci_num = []
    for uci in uci_moves:
        uci_num.append([ord(uci[0])- ord('a'), int(uci[1]) - 1, ord(uci[2]) - ord('a'), int(uci[3]) - 1])
        

    moves = {}
    for uci in uci_moves:
        numerical = np.array([ord(uci[0])- ord('a'), int(uci[1]) - 1, ord(uci[2]) - ord('a'), int(uci[3]) - 1])
        moves[(numerical[0], numerical[1])] = np.array([-1, numerical[2], numerical[3], segment_id + 1])


    counts = defaultdict(lambda: 0)
    #board position is j, i
    #White biship is index 5
    for i in range(8):
        for j in range(8):
            piece = int(board[i][j])
            if piece != 0:
                base = piece_index[str(piece)]

                if abs(piece) != 3:
                    ind = base + counts[piece]
                else:
                    ind = base + isBlack(j,i)

                if (j,i) in moves:
                    moves[(j,i)][0] = ind
                counts[piece] += 1

    out = []
    for i in range(len(uci_num)):
        out.append(moves[(uci_num[i][0], uci_num[i][1])])

    out = np.vstack(out)
    return out

def array_to_bag(array, piece_index, segment_id):
    board = array[:64]
    additional = array[64:69]
    move = array[-4:]
    board = board.reshape((8,8))

    bag = []  
    mv_arr = np.array([-1, move[2], move[3], segment_id + 1])
    counts = defaultdict(lambda: 0)

    #board position is j, i
    #White biship is index 5
    for i in range(8):
        for j in range(8):
            piece = int(board[i][j])
            if piece != 0:
                base = piece_index[str(piece)]

                if abs(piece) != 3:
                    ind = base + counts[piece]
                else:
                    ind = base + isBlack(j,i)

                bag.append(np.array([ind, j, i, segment_id]))

                if j == move[0] and i == move[1]:
                    mv_arr[0] = ind

                counts[piece] += 1

    bag.append(mv_arr)
    bag = np.vstack(bag)
    return bag, additional

#for inference only
#returns bag format and context moves 
def fen_to_bag(fen, encoder, index, k, piece_index, uci = None):
    board = chess.Board(fen)
    
    arr = board_to_array(board)
    arr = np.concatenate((arr, np.zeros(4)))
    bitboard = np.asarray([board_to_bitboard(board)])

    embedding = np.asarray(encoder.predict_on_batch(bitboard)).squeeze()

    results = index.query(vector=embedding.tolist(), top_k = k, include_metadata = True)
    results = results['matches']

    context = []
    context_moves = []
    for i in range(len(results)):
        fen = results[i]['id']
        move = results[i]['metadata']['move']
        context_moves.append(move)

        context_board = chess.Board(fen = fen)
        context_arr = board_to_array(context_board)
        move_arr = np.array([ord(move[0])- ord('a'), int(move[1]) - 1, ord(move[2]) - ord('a'), int(move[3]) - 1])

        context.append(np.concatenate((context_arr, move_arr)))
    data = np.concatenate((np.vstack(context), arr.reshape((1, -1))), axis=0)

    x = []
    s = 0
    ys = None
    for i in range(len(data)):
        bag, add = array_to_bag(data[i], piece_index, i*2 + 1)
        
        s += len(bag)
        x.append(bag)

        if i != len(data) - 1:
            x.append(np.array([[33, 0,0,0]]))

        if i == len(data) - 1 and uci is not None:
            ys = array_to_y(data[i], piece_index, i*2 + 1, uci)
    
    truth = x[-1][-1]
    y = truth.copy()

    truth[0] = 34
    truth[1] = 0
    truth[2] = 0

    x = np.concatenate(x, axis = 0) # (len_seq, 4)

    return x, ys, context_moves

