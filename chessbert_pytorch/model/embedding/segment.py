import torch.nn as nn


class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=128):
        super().__init__((9+1), embed_size, padding_idx=0)
