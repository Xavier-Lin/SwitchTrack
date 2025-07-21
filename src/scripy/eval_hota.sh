# For SportsMOT:
# gt_folder_path=/data/zelinliu/BEE24/test
# val_map_path=/data/zelinliu/BEE24/test_seqmap.txt
gt_folder_path=/data/zelinliu/sportsmot/val
val_map_path=/data/zelinliu/sportsmot/splits_txt/val.txt
track_results_path=/data/zelinliu/SwitchTrack/results/trackval_dc
# need to change 'gt_val_half.txt' or 'gt.txt'
val_type='{gt_folder}/{seq}/gt/gt.txt'

# command
python ../TrackEval/scripts/run_mot_challenge.py  \
        --SPLIT_TO_EVAL train  \
        --METRICS HOTA CLEAR Identity \
        --GT_FOLDER ${gt_folder_path}   \
        --SEQMAP_FILE ${val_map_path}  \
        --SKIP_SPLIT_FOL True   \
        --TRACKERS_TO_EVAL '' \
        --TRACKER_SUB_FOLDER ''  \
        --USE_PARALLEL True  \
        --NUM_PARALLEL_CORES 8  \
        --PLOT_CURVES False   \
        --TRACKERS_FOLDER  ${track_results_path}  \
        --GT_LOC_FORMA ${val_type}