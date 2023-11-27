import numpy as np
import h5py
import pickle
import chess
import chess.pgn
import tensorflow as tf
import sys
import os

from utils import *
import time
import pinecone

def pgn_to_positions(pgn_file, save_file, length_file, encoder_file, embedding_size = 64):

    game_index = -1
    game_lengths = []
    num_context = 8
    dim = 64 + 5 + 4
    pinecone.init(api_key = '38132697-8f87-4930-a355-376bd93394a3', environment = "us-east4-gcp")
    index = pinecone.Index('chesspos-lichess-embeddings')

    encoder = tf.keras.models.load_model(encoder_file)
    with open(pgn_file, 'r') as f:
        with h5py.File(save_file, 'a') as p:

            if 'embeddings' in p:
                data = p['embeddings']
                size = data.shape[0]
            else:
                data = p.create_dataset(f"embeddings", shape=(0, num_context + 1, dim), maxshape=(None, num_context + 1, dim), chunks=True)
                size = 0

            while True:
                next_game = chess.pgn.read_game(f)
                game_index += 1

                if next_game is None:
                    with open(length_file, 'wb') as p:
                        pickle.dump(game_lengths, p)
                    break
                embeddings = game_embeddings(next_game, game_index, encoder, index, num_context)
                if len(embeddings) > 0:
                    data.resize((size + len(embeddings), num_context + 1, dim))
                    data[-len(embeddings):] = embeddings[:]
            
                    size += len(embeddings)
                    game_lengths.append(len(embeddings))

            print(data.shape)
    return 0

def game_embeddings(game, game_ind, encoder, index, k):
    board = chess.Board()
    pos = []
    bitboards = []
    for move in game.mainline_moves():
        arr = board_to_array(board)
        bitboard = board_to_bitboard(board)
        bitboards.append(bitboard)

        start = move.from_square
        end = move.to_square
        mv_arr = np.array([chess.square_file(start), chess.square_rank(start), chess.square_file(end), chess.square_rank(end)])

        try:
            board.push(move)
        except Exception as e:
            print(f"Exception occurred in game number {game_ind}")
            print(e)
            return pos
        else:
            pos.append(np.concatenate((arr, mv_arr)))


    bitboards = np.asarray(bitboards)
    #s = time.time()
    embeddings = np.asarray(encoder.predict_on_batch(bitboards))
    #print(time.time() - s)
        
    for i in range(len(pos)):
        context = []
        results = index.query(vector=embeddings[i].tolist(), top_k = k, include_metadata = True)
        results = results['matches']

        for j in range(len(results)):
            fen = results[j]['id']
            move = results[j]['metadata']['move']

            context_board = chess.Board(fen = fen)
            context_arr = board_to_array(context_board)
            move_arr = np.array([ord(move[0])- ord('a'), int(move[1]) - 1, ord(move[2]) - ord('a'), int(move[3]) - 1])

            context.append(np.concatenate((context_arr, move_arr)))

        pos[i] = np.concatenate((np.vstack(context), pos[i].reshape((1, -1))), axis = 0)
    return pos

if __name__ == '__main__':
    pgn_file = sys.argv[1]
    save_file = sys.argv[2]
    length_file = sys.argv[3]
    encoder_file = sys.argv[4]
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    pgn_to_positions(pgn_file, save_file, length_file, encoder_file)
