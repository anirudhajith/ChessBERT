#!/bin/bash
export CUDA_AVAILABLE_DEVICES='0'
export TF_CPP_MIN_LOG_LEVEL=2
python3 preprocessing.py 'sample_data/' 'sample_data/' 'sample_data/' 'model_encoder.h5'
