#!/usr/bin/env bash

USE_GPU="0" # specify a GPU
echo CUDA_VISIBLE_DEVICES=${USE_GPU}

HOME_DIR="$(pwd)"
source $1 # load parameters

mkdir -p ${EXP_NAME} # run it
LOG_NAME="log.txt"
echo python ${RUN_ARGUMENTS} 
CUDA_VISIBLE_DEVICES=${USE_GPU} python -u ${RUN_ARGUMENTS} | tee ${EXP_NAME}/${LOG_NAME}
