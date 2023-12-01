import argparse
import json
import random
from stockfish import Stockfish
from dataset.utils import *

import pinecone
import tensorflow as tf

def evaluate(fen, candidate_moves, stockfish):
    stockfish.set_fen_position(fen)
    moves = stockfish.get_top_moves(100)

    stockfish_moves = {}
    for i, move in enumerate(moves):
        stockfish_moves[move['Move']] = i

    best = -1
    for i, move in enumerate(candidate_moves):
        if move in stockfish_moves: 
            best = stockfish_moves[move]
            break

    '''
    print("best " + str(best))
    print(len(moves))
    print("sample " + str(random.randrange(0, len(moves))))
    '''
    if best < 0:
        best = random.randrange(0, len(moves)) 

    return best, random.randrange(0, len(moves))

def eval():
    stockfish = Stockfish("/home/david/Masters/VectorCOS597A/stockfish/stockfish-ubuntu-x86-64-avx2")

    pinecone.init(api_key = '38132697-8f87-4930-a355-376bd93394a3', environment = "us-east4-gcp")
    index = pinecone.Index('chesspos-lichess-embeddings')
    piece_index = json.load(open("dataset/preprocessing/piece_index.json", 'r'))
    encoder = tf.keras.models.load_model("dataset/preprocessing/model_encoder.h5")
    
    trank = 0 
    trank_random = 0
    count = 0

    with open("fens.txt", 'r') as f:
        for i, fen in enumerate(f):
            _, _, context_moves = fen_to_bag(fen, encoder, index, 64, piece_index)
            rank, rank_random = evaluate(fen, context_moves, stockfish)
            trank += rank
            trank_random += rank_random
            count += 1

            if i % 50 == 0:
                print(i)
                print("Avg rank of retrieval: %f" % (trank / count))
                print("Avg rank of random: %f" % (trank_random / count))


    print("Avg rank of retrieval: %f" % (trank / count))
    print("Avg rank of random: %f" % (trank_random / count))


if __name__ == '__main__':
    eval()
