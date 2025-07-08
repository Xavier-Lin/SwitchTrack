# 测试命令
python test.py  \
   --track_thresh 0.2  \
   --pre_thresh 0.5 \
   --new_thresh 0.4 \
   --load_model /data/zelinliu/SwitchTrack/pretrain/dla34_h3_adam_ep50.pth \
   --test_device 3  \
   --num_classes 1  \
   --input_h 608  \
   --input_w 1088  \
   --K 20 \
   --num_head_conv 3 \
   --custom_dataset_img_path /data/zelinliu/BEE24/test  \
   --pre_hm \
   --custom_dataset_ann_path /data/zelinliu/BEE24/annotations/test.json  \
   --hungarian \
   --use_bfl \
   --arch dla34 \
   --fuse
   # --use_center ept45!!! - 46.9 HOTA  43.1 AssA  51.7 DetA  65.2 MOTA  60.8 IDF1