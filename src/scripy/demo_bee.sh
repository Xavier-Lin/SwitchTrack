# ------------------------------------------------------- SportsMOT dataset -------------------------------------------------------------#
# DCTrack
model_path=/data/zelinliu/SwitchTrack/pretrain/dla34_h1bfl_adam_ept45.pth
demo_video_path=/data/zelinliu/BEE24/test/BEE24-36/img1

python demo.py  \
   --demo ${demo_video_path} \
   --track_thresh 0.2  \
   --pre_thresh 0.5 \
   --new_thresh 0.4 \
   --load_model ${model_path} \
   --test_device 1  \
   --num_classes 1 \
   --video_h 608  \
   --video_w 1088  \
   --K 20 \
   --num_head_conv 1 \
   --pre_hm \
   --save_video \
   --show_track_color \
   --use_bfl \
   --hungarian # if no hungarian for DC -- Greedy
   # 
   # --use_center # Center Track
   # --use_sort # SORT track
   # --use_fairmot # FairMOT track
   # --use_center_kf # center_kf track 
   # --use_deepsort # DeepSORT track 
   # --use_byte # BYTE track
   # --use_ocsort
