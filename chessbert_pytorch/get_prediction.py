import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import numpy as np
import json
import h5py
from trainer import CLIPTrainer
from dataset import ChessDataset, collate_fn, utils
from model import ChessBERT, MaskedChessModel


MODEL_PATH = "/home/david/Masters/VectorCOS597A/model.ep6"
K = 4
PIECE_INDEX_PATH = "dataset/preprocessing/piece_index.json"

with open(PIECE_INDEX_PATH, "r") as f:
    piece_index = json.load(f)

print("Loading MaskedChessModel")
model = torch.load(MODEL_PATH)
model.to('cuda')
model.eval()

def get_prediction(current_fen, legal_move_ucis, encoder, index) -> int:
    """
    :param current_fen: current board state
    :param legal_move_ucis: list of legal moves
    :param encoder: encoder for converting FEN to board state
    :param index: index for converting FEN to board state
    :return: index of the best move among legal moves
    """
    x, ys, _ = utils.fen_to_bag(current_fen, encoder, index, K, piece_index, legal_move_ucis)
    x = torch.from_numpy(x).unsqueeze(0)
    x = x.to("cuda")
    ys= torch.from_numpy(ys).to("cuda") # (num_legal_moves, 4)
    options = model.chessbert.embedding(ys.unsqueeze(0).to(torch.long)).squeeze(0) # (num_legal_moves, hidden)
    model_prediction = model.forward(x) # (1, hidden)
    logits = torch.mm(model_prediction, options.t()) # (1, num_legal_moves)
    return torch.argmax(logits, dim=1).item(), torch.argsort(logits, dim=1).squeeze().numpy()

def get_prediction_model(current_fen, legal_move_ucis, encoder, index, model, K=4) -> int:
    """
    :param current_fen: current board state
    :param legal_move_ucis: list of legal moves
    :param encoder: encoder for converting FEN to board state
    :param index: index for converting FEN to board state
    :return: index of the best move among legal moves
    """
    x, ys, _ = utils.fen_to_bag(current_fen, encoder, index, K, piece_index, legal_move_ucis)
    x = torch.from_numpy(x).unsqueeze(0)
    x = x.to("cuda")
    ys= torch.from_numpy(ys).to("cuda") # (num_legal_moves, 4)
    options = model.chessbert.embedding(ys.unsqueeze(0).to(torch.long)).squeeze(0) # (num_legal_moves, hidden)
    model_prediction = model.forward(x) # (1, hidden)
    logits = torch.mm(model_prediction, options.t()) # (1, num_legal_moves)
    return torch.argmax(logits, dim=1).item(), torch.argsort(logits, dim=1).squeeze().cpu().numpy()
