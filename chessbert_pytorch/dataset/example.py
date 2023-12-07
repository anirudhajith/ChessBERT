import torch

from torch.utils.data.dataloader import DataLoader
from dataset.chess_dataset import ChessDataset
from dataset.utils import *

data = ChessDataset("dataset/preprocessing/data/data.hdf5", "dataset/preprocessing/piece_index.json", return_context = False)
train_loader = DataLoader(data, batch_size = 8, shuffle=True, collate_fn = collate_fn)

for i, (x, add, y) in enumerate(train_loader):
    print(x.shape) #(batch_size, len_seq, 4)
    print(add.shape) # (batch_size, 9, 5)
    print(y.shape) # (batch_size, 4)

