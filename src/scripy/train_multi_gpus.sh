# Multi GPU training 
CUDA_VISIBLE_DEVICES=0,1,2,3  python -m torch.distributed.launch --nproc_per_node=4  main.py \
   --same_aug_pre   --hm_disturb 0.05  --lost_disturb 0.4   --fp_disturb 0.1  --num_epochs 70 \
   --num_workers 4  --batch_size 48    --input_h 608        --input_w 1088    --num_classes 1  \
   --K 200          --lr 4e-4          --pre_hm             --lr_step '20,40,50' \
   --custom_dataset_img_path  .../UAVMOT/img/train   \
   --custom_dataset_ann_path  .../UAVMOT/labels_with_ids/train.json     


# Multi GPU resume training 
# add --load_model ...  --resume

CUDA_VISIBLE_DEVICES=0,1  python -m torch.distributed.launch --nproc_per_node=2  main.py \
   --same_aug_pre   --hm_disturb 0.05  --lost_disturb 0.4   --fp_disturb 0.1  --num_epochs 90 \
   --num_workers 2  --batch_size 24    --input_h 608        --input_w 1088    --num_classes 1  \
   --K 200          --lr 1e-4          --pre_hm             --lr_step '45' \
   --custom_dataset_img_path  /data/zelinliu/sportsmot/train   \
   --custom_dataset_ann_path  /data/zelinliu/sportsmot/annotations/train.json   \
   --load_model /data/zelinliu/DCtrack/exp1/tracking.ctdet/mot_exp/model_70.pth  --resume


CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1  main.py \
   --same_aug_pre   --hm_disturb 0.05  --lost_disturb 0.4   --fp_disturb 0.1  --num_epochs 70 \
   --num_workers 1  --batch_size 12    --input_h 608        --input_w 1088    --num_classes 1  \
   --K 200          --lr 1e-4          --pre_hm             --lr_step '45' \
   --custom_dataset_img_path  /data/zelinliu/sportsmot/train   \
   --custom_dataset_ann_path  /data/zelinliu/sportsmot/annotations/train.json   