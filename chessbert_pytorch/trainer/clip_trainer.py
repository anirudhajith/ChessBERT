import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from ..model import MaskedChessModel, ChessBERT
from .optim_schedule import ScheduledOptim

import tqdm


class CLIPTrainer:
    """
    CLIPTrainer uses a CLIP objective to train the MaskedChessModel.

    """

    def __init__(self, model: MaskedChessModel, 
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 5e-5, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10):
        """
        :param model: MaskedChessModel which you want to train
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for ChessBERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        
        # Move model to GPU
        self.model = model.to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for MaskedChessModel" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.model.chessbert.hidden, n_warmup_steps=warmup_steps)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.loss_input_embeddings = nn.CrossEntropyLoss()
        self.loss_output_embeddings = nn.CrossEntropyLoss()

        self.log_freq = log_freq

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.model.train()
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.model.eval()
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        str_code = "train" if train else "test"

        avg_loss = 0.0
        for i, batch in enumerate(tqdm(data_loader)):
            x, y = batch    # (batch_size, seq_len, 4), (batch_size, 4)
            x, y = x.to(self.device), y.to(self.device)

            input_embeddings = self.model.chessbert.embedding(y.unsqueeze(1).to(torch.long)).squeeze(1) # (batch_size, hidden)
            output_embeddings = self.model(x) # (batch_size, hidden)
            ground_truth = torch.arange(len(x), dtype=torch.long, device=device)

            total_loss = (self.loss_input_embeddings(input_embeddings, ground_truth) + self.loss_output_embeddings(output_embeddings, ground_truth)) / 2

            if train:
                self.optim_schedule.zero_grad()
                total_loss.backward()
                self.optim_schedule.step_and_update_lr()
            
            avg_loss += total_loss.item()
            
            # logging information.
            if train and i % self.log_freq == 0:
                print("EP%d_%s:%d/%d, loss=%f" % (epoch, str_code, i, len(data_loader), avg_loss / (i + 1)))
            
        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_loader))
        

    def save(self, epoch, file_path="output/masked_chess_model_trained.model"):
        """
        Saving the current MaskedChessModel model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.model.cpu(), output_path)
        self.model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path
