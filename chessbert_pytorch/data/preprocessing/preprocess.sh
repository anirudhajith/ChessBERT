#!/bin/bash
export CUDA_AVAILABLE_DEVICES=''
export TF_CPP_MIN_LOG_LEVEL=2
python3 preprocessing.py 'sample_data/subset.pgn' 'sample_data/test.hdf5' 'sample_data/len.pkl' 'model_encoder.h5'
