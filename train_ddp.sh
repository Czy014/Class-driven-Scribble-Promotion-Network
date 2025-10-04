#!/bin/bash

# DDP 多卡训练启动脚本
# 使用方法: bash train_ddp.sh

# 设置GPU数量
export CUDA_VISIBLE_DEVICES=0,1,2
NUM_GPUS=3

# 基本训练参数
DATASET_PATH="dataset/ScribbleSup/VOC2012"
DATASET="ScribblePseudoDsDc"
MODEL_TYPE="res50_ASPP_lorm"
LAYERS=50
NUM_CLASSES=21
BATCH_SIZE=8  # 每GPU的batch size，总batch size = NUM_GPUS * BATCH_SIZE
EPOCHS=50
LR=1e-3
WORKERS=4

# 损失权重
L_SEG=1
L_PESUDO=1
L_LORM=1
L_DS=1
L_DC=1

# 路径配置
TRAIN_PATH="pascal_2012_scribble"
DISTANCEMAP_S="distance_map"
DISTANCEMAP_C="distance_pseudo"
LOGDIR="./log/train_ddp_r50_deeplabv2"

# 启动分布式训练
python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    --use_env \
    train.py \
    --distributed \
    --layers $LAYERS \
    --model_type $MODEL_TYPE \
    --dataset_path $DATASET_PATH \
    --dataset $DATASET \
    --train_path $TRAIN_PATH \
    --distancemap_s $DISTANCEMAP_S \
    --distancemap_c $DISTANCEMAP_C \
    --l_segs $L_SEG \
    --l_pesudo $L_PESUDO \
    --l_lorm $L_LORM \
    --l_ds $L_DS \
    --l_dc $L_DC \
    --numclasses $NUM_CLASSES \
    --workers $WORKERS \
    --batchsize $BATCH_SIZE \
    --lr $LR \
    --epochs $EPOCHS \
    --logdir $LOGDIR