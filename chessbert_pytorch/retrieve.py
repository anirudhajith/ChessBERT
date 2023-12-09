import argparse
import json
import random
from stockfish import Stockfish
import chess
from dataset.utils import *
from get_prediction import get_prediction_model

import pinecone
import tensorflow as tf
import torch

def evaluate(fen, candidate_moves, stockfish, k_levels):
    stockfish.set_fen_position(fen)
    moves = stockfish.get_top_moves(100)

    stockfish_moves = {}
    for i, move in enumerate(moves):
        stockfish_moves[move['Move']] = i

    legal_moves = []
    random_moves  = []
    for i, move in enumerate(candidate_moves):
        if move in stockfish_moves: 
            legal_moves.append(stockfish_moves[move])
            random_moves.append(random.randrange(0, len(moves)))

    while len(legal_moves) < k_levels[-1]:
        legal_moves.append(random.randrange(0, len(moves)))
        random_moves.append(random.randrange(0, len(moves)))

    ranks = np.zeros(len(k_levels))
    rand_ranks = np.zeros(len(k_levels))
    recall = np.zeros(len(k_levels))
    rand_recall = np.zeros(len(k_levels))

    min_r = 101
    min_rand = 101
    c = 0
    for i, k in enumerate(k_levels):
        while c < k:
            min_r = min(min_r, legal_moves[c])
            min_rand = min(min_rand, random_moves[c])
            
            c += 1
        ranks[i] = min_r
        rand_ranks[i] = min_rand
        
        if min_r == 0:
            recall[i] = 1
        if min_rand == 0:
            rand_recall[i] = 1


    return ranks, rand_ranks, recall, rand_recall
    '''
    print("best " + str(best))
    print(len(moves))
    print("sample " + str(random.randrange(0, len(moves))))
    '''

def eval():
    stockfish = Stockfish("/home/david/Masters/VectorCOS597A/stockfish/stockfish-ubuntu-x86-64-avx2")

    index = pinecone.Index('chesspos-lichess-embeddings')
    piece_index = json.load(open("dataset/preprocessing/piece_index.json", 'r'))
    encoder = tf.keras.models.load_model("dataset/preprocessing/model_encoder.h5")


    MODEL_PATH = "/home/david/Masters/VectorCOS597A/no_context_model.ep6"
    K = 4
    print("Loading MaskedChessModel")
    model = torch.load(MODEL_PATH)
    model.to('cuda')
    model.eval()

    k_levels = [1,3,5,10]
    
    trank = np.zeros(len(k_levels)) 
    trank_random = np.zeros(len(k_levels))
    trecall = np.zeros(len(k_levels))
    trecall_random = np.zeros(len(k_levels))
    count = 0

    with open("fens.txt", 'r') as f:
        for i, fen in enumerate(f):
            board = chess.Board(fen=fen)
            legal_moves = [move for move in board.legal_moves]
            uci_moves = [move.uci() for move in legal_moves]
            _, move_indexes = get_prediction_model(board.fen(), uci_moves, encoder, index, model)

            real_moves = []
            
            if len(move_indexes) > 0:
                for j in range(len(move_indexes)):
                    real_moves.append(uci_moves[move_indexes[j]])

                rank, rank_random, recall, recall_random = evaluate(fen, real_moves, stockfish, k_levels)
                trank += rank
                trank_random += rank_random
                trecall += recall
                trecall_random += recall_random
                print(i)
                count += 1

                if i % 50 == 0:
                    print(i)
                    print("Avg rank of retrieval: %s" % str(trank / count))
                    print("Avg rank of random: %s" % str(trank_random / count))
                    print("Avg recall of retrieval: %s" % str(trecall / count))
                    print("Avg recall of random: %s" % str(trecall_random / count))


    print("Avg rank of retrieval: %s" % str(trank / count))
    print("Avg rank of random: %s" % str(trank_random / count))
    print("Avg recall of retrieval: %s" % str(trecall / count))
    print("Avg recall of random: %s" % str(trecall_random / count))



if __name__ == '__main__':
    eval()
