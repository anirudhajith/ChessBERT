from torch.utils.data import Dataset
import json
from .utils import *
import h5py

class ChessDataset(Dataset):
    def __init__(self, position_file, piece_index_file, return_context = True, sep = 33, mask = 34):
        with open(piece_index_file, 'r') as f:
            self.piece_index = json.load(f)
    
        f = h5py.File(position_file, 'r')
        self.dset = f['embeddings']

        self.sep_token = np.array([[sep, 0, 0, 0]])
        self.mask = mask
        self.return_context = return_context
        
    def __len__(self):
        return self.dset.shape[0]

    def __getitem__(self, ind):
        data = self.dset[ind]
         
        x = []
        rights = []
        s = 0
        for i in range(len(data)):
            if return_context or i == len(data) - 1:
                bag, add = array_to_bag(data[i], self.piece_index, i*2 + 1)
            
                s += len(bag)
                x.append(bag)
                rights.append(add)

                if i != len(data) - 1:
                    x.append(self.sep_token.copy())
        
        truth = x[-1][-1]
        y = truth.copy()

        truth[0] = self.mask
        truth[1] = 0
        truth[2] = 0

        x = np.concatenate(x, axis = 0) # (len_seq, 4)
        rights = np.vstack(rights) # (9, 5)

        return x, rights, y
