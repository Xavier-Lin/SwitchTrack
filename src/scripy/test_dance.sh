# 测试命令
python test.py  \
   --track_thresh 0.2  \
   --pre_thresh 0.5 \
   --new_thresh 0.9 \
   --load_model /data/zelinliu/SwitchTrack/exp/tracking.ctdet/mot_exp/model_90.pth  \
   --test_device 0  \
   --num_classes 1  \
   --input_h 608  \
   --input_w 1088  \
   --K 50 \
   --num_head_conv 1 \
   --custom_dataset_img_path /data/zelinliu/DanceTrack/dancetrack/val  \
   --pre_hm \
   --custom_dataset_ann_path /data/zelinliu/DanceTrack/dancetrack/annotations/val.json  \
   --hungarian \
   --use_bfl \
   --arch dla60 \
   --max_age 30 \
   --fuse
   # --use_center ept45!!! - 46.9 HOTA  43.1 AssA  51.7 DetA  65.2 MOTA  60.8 IDF1