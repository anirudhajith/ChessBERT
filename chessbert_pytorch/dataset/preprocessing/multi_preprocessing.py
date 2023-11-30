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
import glob
import random
import pinecone
import multiprocessing as mp
from multiprocessing import Process, Queue

def pinecone_query(in_q, out_qs, index, num_context):
    while True:
        queries, batch_num, id = in_q.get()
        if id == None:
            break

        results = None
        while results is None:
            try:
                results = index.query(queries = queries.tolist(), top_k=num_context, include_metadata = True)
                results = results['results']

                out = []
                for i in range(len(results)):
                    matches = results[i]['matches']

                    fens = []
                    moves = []
                    for j in range(len(matches)):
                        fens.append(matches[j]['id'])
                        moves.append(matches[j]['metadata']['move'])

                    out.append([fens, moves])

                out_qs[id].put((out, batch_num))
            except Exception as e:
                print(e)
                continue

def pgn_to_positions(pgn_file, save_file, progress_file, encoder_file, embedding_size, start_index, in_q, out_qs, id):

    game_index = -1
    num_context = 4
    dim = 64 + 5 + 4

    encoder = tf.keras.models.load_model(encoder_file)
    start = time.time()

    with open(pgn_file, 'r') as f:
        with h5py.File(save_file, 'a') as p:

            if 'embeddings' in p:
                data = p['embeddings']
                size = data.shape[0]
            else:
                data = p.create_dataset(f"embeddings", shape=(0, num_context + 1, dim), maxshape=(None, num_context + 1, dim), dtype = np.byte, chunks=True, compression='gzip')
                size = 0

            while True:
                try:
                    next_game = chess.pgn.read_game(f)
                except Exception as e:
                    print(e)
                    continue
                game_index += 1

                if next_game is None:
                    break

                if game_index >= start_index:
                    embeddings = game_embeddings(next_game, game_index, encoder, num_context, in_q, out_qs, id)
                    if len(embeddings) > 0:
                        data.resize((size + len(embeddings), num_context + 1, dim))
                        data[-len(embeddings):] = embeddings[:]
                
                        size += len(embeddings)
                        with open(progress_file, 'w') as tmp:
                            tmp.write(str(game_index))

            print(time.time() - start)
            print(data.shape)
    return 0

def game_embeddings(game, game_ind, encoder, k, in_q, out_qs, id):
    board = chess.Board()
    pos = []
    bitboards = []

    #but why is there no __len__ function????
    l = 0
    for move in game.mainline_moves():
        l+=1

    num_samples = min(60, l)
    sample_inds = set(random.sample(range(l), num_samples))
    for i, move in enumerate(game.mainline_moves()):
        arr = board_to_array(board)
        bitboard = board_to_bitboard(board)

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
            if i in sample_inds:
                pos.append(np.concatenate((arr, mv_arr)))
                bitboards.append(bitboard)
    
    if len(bitboards) == 0:
        return []

    bitboards = np.asarray(bitboards)
    s = time.time()
    embeddings = np.asarray(encoder.predict_on_batch(bitboards))
    
    s = time.time()
    results = [None for i in range(len(embeddings))]
    batch_size = 12
    batch_count = 0
    for i in range(0, len(embeddings), batch_size):
        batch = embeddings[i:i+batch_size]
        in_q.put((batch, batch_count, id))
        batch_count +=1

    for i in range(batch_count):
        result, batch_num = out_qs[id].get()
        si = batch_num * batch_size
        results[si:si+batch_size] = result

    for i in range(len(pos)):
        context = []
        fens = results[i][0]
        moves = results[i][1]

        for j in range(len(fens)):
            fen = fens[j]
            move = moves[j]

            context_board = chess.Board(fen = fen)
            context_arr = board_to_array(context_board)
            move_arr = np.array([ord(move[0])- ord('a'), int(move[1]) - 1, ord(move[2]) - ord('a'), int(move[3]) - 1])

            context.append(np.concatenate((context_arr, move_arr)))

        pos[i] = np.concatenate((np.vstack(context), pos[i].reshape((1, -1))), axis = 0).astype(np.byte)
    return pos

if __name__ == '__main__':
    pgn_dir = sys.argv[1]
    save_dir = sys.argv[2]
    progress_dir = sys.argv[3]
    encoder_file = sys.argv[4]

    pinecone.init(api_key="38132697-8f87-4930-a355-376bd93394a3", environment="us-east4-gcp")
    index = pinecone.Index("chesspos-lichess-embeddings")


    pgn_files = glob.glob("%s/*.pgn" % (pgn_dir))
    pgn_files.sort()

    num_workers = 4
    workers = []
    queries = []
    in_q = mp.Queue()
    out_qs = [mp.Queue() for i in range(num_workers)]

    start = [415, 391, 442, 401]
    #start = [0,0]
    for i in range(num_workers):
        save_file = "%s/%d.hdf5" % (save_dir, i)
        progress_file = "%s/%d.txt" % (progress_dir, i)
        workers.append(Process(target = pgn_to_positions, args = (pgn_files[i], save_file, progress_file, encoder_file, 64, start[i], in_q, out_qs, i)))

    for i in range(10):
        queries.append(Process(target = pinecone_query, args = (in_q, out_qs, index, 4)))

    for worker in workers:
        worker.start()

    for i in range(len(queries)):
        queries[i].start()

    for worker in workers:
        worker.join()
    
    for i in range(len(queries)):
        in_q.put((None, None, None))
    
