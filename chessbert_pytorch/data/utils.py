from preprocessing.utils import *
import numpy as np
import h5py
import pickle
import chess
from collections import defaultdict
import pinecone
import random

from torch.utils.data import default_collate

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

    assert mv_arr[0] != -1
    bag.append(mv_arr)
    bag = np.vstack(bag)
    return bag, additional

