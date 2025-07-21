#!/bin/bash

# 1. 环境设置
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 设置可见的GPU设备 0,1,2,3
export PYTHONPATH=/data/zelinliu/SwitchTrack/src:$PYTHONPATH

# 2. 参数配置
NUM_GPUS=4  # 4
HM_DISTURB=0.05
LOST_DISTURB=0.4
FP_DISTURB=0.1
NUM_EPOCHS=90
NUM_WORKERS=4 # 4
BATCH_SIZE=16 # 24 -- 0.0002
INPUT_H=608
INPUT_W=1088
NUM_CLS=1
NUM_DETS=100
LEARNING_RATE=0.00013
OPTIM="adam"
NUM_H=1
LR_STEP="50"
ARCH="dla60"
DATA_IMG_PATH="/data/zelinliu/DanceTrack/dancetrack/train"
DATA_ANN_PATH="/data/zelinliu/DanceTrack/dancetrack/annotations/train.json"


# 3. 训练命令
python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    main.py \
    --same_aug_pre   \
    --hm_disturb $HM_DISTURB  \
    --lost_disturb $LOST_DISTURB  \
    --fp_disturb $FP_DISTURB  \
    --num_epochs $NUM_EPOCHS   \
    --num_workers $NUM_WORKERS  \
    --batch_size $BATCH_SIZE  \
    --input_h $INPUT_H   \
    --input_w $INPUT_W \
    --num_classes $NUM_CLS \
    --K $NUM_DETS  \
    --lr $LEARNING_RATE  \
    --pre_hm  \
    --optim $OPTIM \
    --num_head_conv $NUM_H \
    --use_bfl \
    --lr_step $LR_STEP \
    --arch $ARCH \
    --custom_dataset_img_path $DATA_IMG_PATH  \
    --custom_dataset_ann_path  $DATA_ANN_PATH