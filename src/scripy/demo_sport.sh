# ------------------------------------------------------- SportsMOT dataset -------------------------------------------------------------#
# DCTrack
model_path=/data/zelinliu/DCtrack/exp/tracking.ctdet/mot_exp/model_70.pth
demo_video_path=/data/zelinliu/sportsmot/val/v_0kUtTtmLaJA_c006/img1

python demo.py  \
   --demo ${demo_video_path} \
   --track_thresh 0.2  \
   --pre_thresh 0.5 \
   --load_model ${model_path} \
   --test_device 1  \
   --num_classes 1 \
   --video_h 608  \
   --video_w 1088  \
   --K 50 \
   --pre_hm \
   --save_video \
   --show_track_color \
   --hungarian # if no hungarian for DC -- Greedy
   # 
   # --use_center # Center Track
   # --use_sort # SORT track
   # --use_fairmot # FairMOT track
   # --use_center_kf # center_kf track 
   # --use_deepsort # DeepSORT track 
   # --use_byte # BYTE track
   # --use_ocsort
