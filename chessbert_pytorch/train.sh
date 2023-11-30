#!/bin/bash
/home/aa8052/.conda/envs/chessbert/bin/python3 . --train_dataset data/train.hdf5 --test_dataset data/test.hdf5 --piece_index dataset/preprocessing/piece_index.json -o .

