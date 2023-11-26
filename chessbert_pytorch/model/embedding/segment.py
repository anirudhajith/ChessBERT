import torch.nn as nn


class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=128):
        super().__init__((1 + 9*2), embed_size, padding_idx=0)
