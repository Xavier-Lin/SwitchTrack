# 测试命令
model_path=/data/zelinliu/SwitchTrack/exp/tracking.ctdet/mot_exp/model_90.pth
demo_video_path=/data/zelinliu/DanceTrack/dancetrack/val/dancetrack0063/img1

python demo.py  \
   --demo ${demo_video_path} \
   --track_thresh 0.2  \
   --pre_thresh 0.5 \
   --new_thresh 0.9 \
   --load_model ${model_path} \
   --test_device 1  \
   --num_classes 1 \
   --video_h 608  \
   --video_w 1088  \
   --K 50 \
   --num_head_conv 1 \
   --arch dla60 \
   --max_age 30 \
   --pre_hm \
   --use_bfl \
   --save_video \
   --show_track_color \
   --hungarian # if no hungarian for DC -- Greedy