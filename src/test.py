from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import cv2
import copy
import torch
import numpy as np
import os.path as osp
import motmetrics as mm

from loguru import logger
from lib.opts import opts
from lib.detector import Detector
from lib.tracking_utils.timer import Timer
from lib.tracking_utils.evaluation import Evaluator
from lib.dataset.dataset_factory import dataset_factory


class SubsectDataset(torch.utils.data.Dataset):
  def __init__(self, opt, dataset, pre_process_func, seq, ig_id, seq_num):
    self.images = dataset.images[ig_id:seq_num+ig_id] # all img_ids for dataset
    self.load_image_func = dataset.coco.loadImgs # self.coco.loadImgs
    self.img_dir = dataset.img_dir # seqs path
    self.pre_process_func = pre_process_func 
    self.get_default_calib = dataset.get_default_calib 
    self.opt = opt
  
  def __getitem__(self, index):
    img_id = self.images[index]
    img_info = self.load_image_func(ids=[img_id])[0]
    img_path = os.path.join(self.img_dir, img_info['file_name'])
    image = cv2.imread(img_path)
    images, meta = {}, {}
    for scale in opt.test_scales:
      input_meta = {}
      calib = img_info['calib'] if 'calib' in img_info \
        else self.get_default_calib(image.shape[1], image.shape[0])
      input_meta['calib'] = calib # get calib
      images[scale], meta[scale] = self.pre_process_func(
        image, scale, input_meta)#input img:tensor  meta
    ret = {'images': images, 'image': image, 'meta': meta}
    # ret = {'images':input img:c h w, 'image':original img, 'meta': img relevant data}
    if 'frame_id' in img_info and img_info['frame_id'] == 1:
      ret['is_first_frame'] = 1
      ret['video_id'] = img_info['video_id']
    return img_id, ret 
    # img_id , ret = {
    #           'images':input img:c h w, 'image':original img, 'meta': img relevant data,
    #           'is_first_frame':1 if a img is the first frame of video seq,'video_id': the video id }

  def __len__(self):
    return len(self.images)

def mkdir_if_missing(d):
    if not osp.exists(d):
        os.makedirs(d)

def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))

def xyxy2x1y1wh(bbox):
    box=copy.deepcopy(bbox)
    box[2] = box[2] - box[0]
    box[3] = box[3] - box[1]
    return box.tolist()

def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, 
                trackers = None, seq_num = None):
    if save_dir:
        mkdir_if_missing(save_dir)
    
    timer = Timer()
    results = []
    frame_id = 0
    trackers.reset_tracking()

    for ind, (img_id, pre_processed_images) in enumerate(dataloader):
        if ind >= seq_num:
            break
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        if opt.tracking and ('is_first_frame' in pre_processed_images):
            pre_processed_images['meta']['pre_dets'] = []

        timer.tic()
        ret = trackers.run(pre_processed_images)
        timer.toc()
        online_tlwhs = []
        online_ids = []
        for t in ret['results']:
            if opt.use_fairmot or opt.use_center_kf or opt.use_byte:
                tlwh = t.tlwh # right
                tid = t.track_id
            elif opt.use_sort or opt.use_deepsort or opt.use_ocsort:
                tlwh = np.array(xyxy2x1y1wh(t[:4]), dtype=np.float32)
                tid = int(t[4])
            elif opt.use_center:
                tlwh = [t['bbox'][0], t['bbox'][1], t['bbox'][2] - t['bbox'][0],  t['bbox'][3] - t['bbox'][1]]
                tid = t['tracking_id'] 
            else:
                tlwh = xyxy2x1y1wh(t.tlbr)
                tid = t.track_id
                
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
        
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        frame_id += 1

    # association_by_emb = trackers.tracker.num_emb
    # logger.info('The number of objects assoicated by embedding : {}'.format(association_by_emb))
    # save results
    write_results(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls, img_id


def main(opt, data_root='', seqs=('MOT16-05',), exp_name='demo', show_image=True):
    torch.cuda.set_device(opt.test_device)
    
    # setting result path
    result_root = os.path.join(os.path.dirname(__file__), '..', 'results', exp_name)
    mkdir_if_missing(result_root)
   
    data_type = 'mot'
    Dataset = dataset_factory[opt.test_dataset]
    
    # setting logger
    LOG_PATH = osp.join(opt.save_dir, 'tracking_log')
    mkdir_if_missing(LOG_PATH)
    logger.add(osp.join(LOG_PATH, 'tracking_{time}.log'))
    logger.info(opt)
    
    split = 'val' 
    dataset = Dataset(opt, split)
    build_tracker = Detector(opt)

    # run tracking
    n_frame = 0
    ig_id = 0
    accs = []
    timer_avgs, timer_calls = [], []
    for seq in seqs:
        logger.info('start seq: {}'.format(seq))
        # import pdb;pdb.set_trace()
        seq_num=len([i for i in  os.listdir(osp.join(opt.custom_dataset_img_path, seq, 'img1' if not('UAVDT' in data_root) else ''))if 'jpg' in i])
        data_loader = torch.utils.data.DataLoader(
                        SubsectDataset(opt, dataset, build_tracker.pre_process, seq, ig_id, seq_num), 
                        batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        nf, ta, tc, ig_id = eval_seq(opt, data_loader, data_type, result_filename,
                               trackers=build_tracker, seq_num=seq_num)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(opt.custom_dataset_img_path, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))
    
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    logger.info('\n'+strsummary)


if __name__ == '__main__':
    opt = opts().init()
    data_root = opt.test_data_root
    seqs_path = opt.custom_dataset_img_path
    seqs_str = [s for s in os.listdir(seqs_path) if not ('.D' in s)]
    seqs = sorted([seq for seq in seqs_str])
    val_name = 'dc' if opt.hungarian else ('ct' if opt.use_center else ('ct_kf' if opt.use_center_kf else ('fair' if opt.use_fairmot else ('sort' if opt.use_sort else ('ocsort' if opt.use_ocsort else ('deepsort' if opt.use_deepsort else 'byte'))))))

    main(opt,
         data_root=data_root,
         seqs=seqs,
         exp_name=f'trackval_{val_name}',
         show_image=False,
        )