#!/bin/bash
export CUDA_AVAILABLE_DEVICES=''
export TF_CPP_MIN_LOG_LEVEL=2
python3 preprocessing.py 'data/' 'data/' 'data/' 'model_encoder.h5' > logs 2>&1 
