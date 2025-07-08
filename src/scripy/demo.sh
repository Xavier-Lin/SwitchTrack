#DCtrack demo
python demo.py  --demo .../UAVMOT/img/test/M0203   --test_device 0 --num_classes 1  \
        --load_model ../exp_with_heatmap/tracking.ctdet/mot_drone_exp/model_70.pth --K 200 --pre_hm \
        --track_thresh 0.3  --pre_thresh 0.55  --save_video --hungarian \
        --video_h 544  --video_w 1024

#FairMOT demo
python demo.py  --demo .../UAVMOT/img/test/M0203   --test_device 0 --num_classes 1  \
        --load_model ../exp_with_heatmap/tracking.ctdet/mot_drone_exp/model_65.pth --K 200 --pre_hm \
        --track_thresh 0.3  --pre_thresh 0.55  --save_video --use_fairmot \
        --video_h 544  --video_w 1024

#DeepSORT demo
python demo.py  --demo .../UAVMOT/img/test/M0203   --test_device 1 --num_classes 1  \
        --load_model ../exp_with_heatmap/tracking.ctdet/mot_drone_exp/model_70.pth --K 200 --pre_hm \
        --track_thresh 0.3  --pre_thresh 0.55  --save_video --use_deepsort \
        --video_h 544  --video_w 1024


#BYTE demo
python demo.py  --demo .../UAVMOT/img/test/M0203   --test_device 0 --num_classes 1  \
        --load_model ../exp_with_heatmap/tracking.ctdet/mot_drone_exp/model_65.pth --K 200 --pre_hm \
        --track_thresh 0.3  --pre_thresh 0.55  --save_video --use_byte \
        --video_h 544  --video_w 1024