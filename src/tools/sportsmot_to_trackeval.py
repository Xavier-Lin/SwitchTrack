import os
import shutil

TRACKEVAL_DATA_ROOT = "ROOTDIR/trackeval"
SPORTSMOT_DATA_ROOT = "ROOTDIR/sportsmot"
NO_SUBDIR = True

for split in ["train", "val", "test"]:
    outputs_dir = f"ROOTDIR/results/{split}"
    mybenchmark_name = "sportsmot"
    split_txt = f"ROOTDIR/splits_txt/{split}.txt"
    tracker_name = "tracker_to_eval"

    # ========default hierarchy========
    # folder hierarchy according to
    # https://github.com/JonathonLuiten/TrackEval/blob/master/docs/MOTChallenge-Official/Readme.md#evaluating-on-your-own-data
    eval_gt_dir = os.path.join(
        TRACKEVAL_DATA_ROOT, "gt" if NO_SUBDIR else "data/gt/mot_challenge",
        mybenchmark_name + "-" + split)
    os.makedirs(eval_gt_dir, exist_ok=True)
    eval_seqmap_dir = os.path.join(
        TRACKEVAL_DATA_ROOT, "gt" if NO_SUBDIR else "data/gt/mot_challenge",
        "seqmaps")
    os.makedirs(eval_seqmap_dir, exist_ok=True)
    eval_trackers_dir = os.path.join(
        TRACKEVAL_DATA_ROOT,
        "trackers" if NO_SUBDIR else "data/trackers/mot_challenge",
        mybenchmark_name + "-" + split, tracker_name, "data")
    os.makedirs(eval_trackers_dir, exist_ok=True)

    shutil.copy(
        split_txt,
        os.path.join(eval_seqmap_dir, f"{mybenchmark_name}-{split}.txt"))
    with open(split_txt, "r", encoding="utf-8") as f:
        for line in f.readlines():
            # copy gt/ and seqinfo.ini
            video_name = line.strip()
            source_dir = os.path.join(SPORTSMOT_DATA_ROOT, split, video_name)
            target_dir = os.path.join(eval_gt_dir, video_name)

            source_gt_dir = os.path.join(source_dir, "gt")
            target_gt_dir = os.path.join(target_dir, "gt")
            shutil.copytree(source_gt_dir, target_gt_dir)
            source_seqinfo = os.path.join(source_dir, "seqinfo.ini")
            shutil.copy(source_seqinfo, target_dir)

            # copy tracker output
            source_track = os.path.join(outputs_dir, f"{video_name}.txt")
            assert os.path.isfile(source_track)
            shutil.copy(source_track, eval_trackers_dir)
