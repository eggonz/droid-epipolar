#!/bin/bash

source activate droidenv

cd $HOME/droidslam-epipolar

DATA_ROOT=""
DATAPATH=$DATA_ROOT/epipolar_data/datasets/TartanAir
WANDBKEY=""
WANDBUSR=""

EXPNAME=train_$(date +'%Y%m%dT%H%M%S')
python train.py --name=$EXPNAME --datapath=$DATAPATH --gpus=1 --lr=0.00025 --steps=30002 \
    --wandb --wandb_entity=$WANDBUSR --wandb_key=$WANDBKEY --wandb_project=droidslam_epipolar
