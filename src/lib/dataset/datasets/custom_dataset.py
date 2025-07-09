from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from collections import defaultdict
import numpy as np
from ..generic_dataset import GenericDataset
from loguru import logger as lg

class CustomDataset(GenericDataset):
  num_categories = 1
  class_name = ['']
  max_objs = 0
  cat_ids = {1: 1}
  def __init__(self, opt, split):
    # import pdb
    # pdb.set_trace()
    self.max_objs = opt.K
    img_dir = opt.custom_dataset_img_path
    ann_path = opt.custom_dataset_ann_path
    self.num_categories = opt.num_classes
    self.default_resolution = [opt.input_h, opt.input_w]
    self.cat_ids = {i: i for i in range(1, self.num_categories + 1)}
    self.images = None
    # load image list and coco
    super().__init__(opt, split, ann_path, img_dir)

    self.num_samples = len(self.images)
    if opt.local_rank in [-1,0]:
      lg.info('Loaded Custom dataset {} samples'.format(self.num_samples))
  
  def __len__(self):
    return self.num_samples

  def save_results(self, results, save_dir):
    # import pdb
    # pdb.set_trace()
    results_dir = os.path.join(save_dir, 'results_mot')
    if not os.path.exists(results_dir):
      os.mkdir(results_dir)
    for video in self.coco.dataset['videos']:
      video_id = video['id']
      file_name = video['file_name']
      out_path = os.path.join(results_dir, '{}.txt'.format(file_name))
      f = open(out_path, 'w')
      images = self.video_to_images[video_id]
      tracks = defaultdict(list)
      for image_info in images:
        if not (image_info['id'] in results):
          continue
        result = results[image_info['id']]
        frame_id = image_info['frame_id']
        for item in result:
          if not ('tracking_id' in item):
            item['tracking_id'] = np.random.randint(100000)
          if item['active'] == 0:
            continue
          tracking_id = item['tracking_id']
          bbox = item['bbox']
          bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
          tracks[tracking_id].append([frame_id] + bbox)
      rename_track_id = 0
      for track_id in sorted(tracks):
        rename_track_id += 1
        for t in tracks[track_id]:
          f.write('{},{},{:.2f},{:.2f},{:.2f},{:.2f},1,1,-1\n'.format(
            t[0], rename_track_id, t[1], t[2], t[3]-t[1], t[4]-t[2]))
      f.close()

  def run_eval(self, results, save_dir):
    # import pdb
    # pdb.set_trace()
    self.save_results(results, save_dir)
    
    # gt_type_str = ''
    # os.system('python tools/eval_motchallenge.py ' + \
    #           '{}'.format('/home/zelinliu/MOT_Drone/test/sequences/') + \
    #           '{}/results_mot{}/ '.format(save_dir, self.dataset_version) + \
    #           gt_type_str + ' --eval_official')
