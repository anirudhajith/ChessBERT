import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import numpy as np
import h5py
from trainer import CLIPTrainer
from dataset import ChessDataset, collate_fn
from model import ChessBERT, MaskedChessModel


MODEL_PATH = "/scratch/gpfs/aa8052/ChessBERT/chessbert_pytorch/..ep6"
TRAIN_DATA_PATH = "/scratch/gpfs/aa8052/ChessBERT/chessbert_pytorch/data/train.hdf5"
TEST_DATA_PATH = "/scratch/gpfs/aa8052/ChessBERT/chessbert_pytorch/data/test.hdf5"
PIECE_INDEX_PATH = "/scratch/gpfs/aa8052/ChessBERT/chessbert_pytorch/dataset/preprocessing/piece_index.json"
BATCH_SIZE = 64
NUM_WORKERS = 5
HIDDEN = 256
LAYERS = 8
ATTN_HEADS = 8
LR = 1e-3
ADAM_WEIGHT_DECAY = 0.01
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
LOG_FREQ = 10
WITH_CUDA = True
CUDA_DEVICES = None


print("Loading Train Dataset", TRAIN_DATA_PATH)
train_dataset = ChessDataset(TRAIN_DATA_PATH, PIECE_INDEX_PATH)

print("Loading Test Dataset", TEST_DATA_PATH)
test_dataset = ChessDataset(TEST_DATA_PATH, PIECE_INDEX_PATH)

print("Creating Dataloader")
train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, collate_fn = collate_fn)
test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, collate_fn = collate_fn) \
    if test_dataset is not None else None

print("Loading MaskedChessModel")
model = torch.load(MODEL_PATH)
model.eval()

print("Creating BERT Trainer")
trainer = CLIPTrainer(model, train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                        lr=LR, betas=(ADAM_BETA1, ADAM_BETA2), weight_decay=ADAM_WEIGHT_DECAY,
                        with_cuda=WITH_CUDA, cuda_devices=CUDA_DEVICES, log_freq=LOG_FREQ)

trainer.iteration(1, test_data_loader, train=False)


