from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import cv2
import os
from collections import defaultdict
import os.path as osp
import pycocotools.coco as coco
import torch
import torch.utils.data as data
from collections import OrderedDict
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian
import copy
from loguru import logger as lg
import pdb

class GenericDataset(data.Dataset):
  is_fusion_dataset = False
  num_categories = None
  class_name = None
  # cat_ids: map from 'category_id' in the annotation files to 1..num_categories
  # Not using 0 because 0 is used for don't care region and ignore loss.
  cat_ids = None
  max_objs = None
  rest_focal_length = 1200
  num_joints = 17
  flip_idx = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], 
              [11, 12], [13, 14], [15, 16]]
  edges = [[0, 1], [0, 2], [1, 3], [2, 4], 
           [4, 6], [3, 5], [5, 6], 
           [5, 7], [7, 9], [6, 8], [8, 10], 
           [6, 12], [5, 11], [11, 12], 
           [12, 14], [14, 16], [11, 13], [13, 15]]
  mean = np.array([0.40789654, 0.44719302, 0.47026115],
                   dtype=np.float32).reshape(1, 1, 3)
  std  = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)
  _eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                      dtype=np.float32)
  _eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
  ignore_val = 1
  def __init__(self, opt=None, split=None, ann_path=None, img_dir=None):
    super(GenericDataset, self).__init__()
    if opt is not None and split is not None:
      self.split = split
      self.opt = opt
      self._data_rng = np.random.RandomState(123)
    
    if ann_path is not None and img_dir is not None:
      lg.info('==> initializing {} data from {}, \n images from {} ...'.format(split, ann_path, img_dir))
      self.coco = coco.COCO(ann_path)
      self.images = self.coco.getImgIds()
      # maxid = np.max(self.images)
      # minid = np.min(self.images)
      if opt.tracking:
        if not ('videos' in self.coco.dataset):
          self.fake_video_data()
        lg.info('Creating video index!')
        self.video_to_images = defaultdict(list)
        for image in self.coco.dataset['images']:
          if '.D' in image['file_name']:
            continue
          self.video_to_images[image['video_id']].append(image)
      
      self.img_dir = img_dir

      if not('test' in ann_path):
        if opt.reid:
          all_video_name = os.listdir(img_dir)
          all_id = 0
          for video_name in all_video_name:
            if '.D' in video_name:
              continue
            video_ann_path = osp.join(img_dir, video_name, 'gt', 'gt.txt')
            video_ann = np.loadtxt(video_ann_path, dtype=np.float32, delimiter=',')
            max_track_id_per_video = np.max(video_ann[:, 1])
            min_track_id_per_video = np.min(video_ann[:, 1])
            all_id += (int(max_track_id_per_video) + 1) #if int(min_track_id_per_video) == 0 else int(max_track_id_per_video)
          opt.nid = all_id
        lg.info('{} dataset the number of track id is : {} '.format(split, opt.nid))
        

  def __getitem__(self, index):
    opt = self.opt
    img, anns, img_info, img_path = self._load_data(index)
   
    #vis
    # im=img.copy()
    # for a in anns:
    #   bbox=a['bbox']
    #   im=cv2.rectangle(im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]+bbox[0]-1), int(bbox[3]+bbox[1]-1)), color=(0, 0, 255), thickness=1)
    # cv2.imwrite('./ddd1.jpg', im)
     
    height, width = img.shape[0], img.shape[1]
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    # c = np.array([w/2, h/2]) 
    s = max(img.shape[0], img.shape[1]) * 1.0 if not self.opt.not_max_crop \
      else np.array([img.shape[1], img.shape[0]], np.float32)
  
    aug_s, rot, flipped = 1, 0, 0
    if self.split == 'train':
      c, aug_s, rot = self._get_aug_param(c, s, width, height)
      #c = np.array([w/2, h/2])
      s = s * aug_s
      if np.random.random() < opt.flip:
        flipped = 1
        img = img[:, ::-1, :]
        anns = self._flip_anns(anns, width)

    trans_input = get_affine_transform(
      c, s, rot, [opt.input_w, opt.input_h])
    trans_output = get_affine_transform(
      c, s, rot, [opt.output_w, opt.output_h])
    inp = self._get_input(img, trans_input)
    ret = {'image': inp}

    pre_cts, track_ids = None, None
    if opt.tracking:
      pre_image, pre_anns, frame_dist = self._load_pre_data(
        img_info['video_id'], img_info['frame_id'], 
        img_info['sensor_id'] if 'sensor_id' in img_info else 1) 
        
      if flipped:
        pre_image = pre_image[:, ::-1, :].copy()
        pre_anns = self._flip_anns(pre_anns, width)
      if opt.same_aug_pre and frame_dist != 0:
        trans_input_pre = trans_input 
        trans_output_pre = trans_output
      else:
        c_pre, aug_s_pre, _ = self._get_aug_param(
          c, s, width, height, disturb=True)
        s_pre = s * aug_s_pre#same as curr frame aug using
        trans_input_pre = get_affine_transform(
          c_pre, s_pre, rot, [opt.input_w, opt.input_h])
        trans_output_pre = get_affine_transform(
          c_pre, s_pre, rot, [opt.output_w, opt.output_h])
          

      pre_img = self._get_input(pre_image, trans_input_pre)# c h w
      pre_hm, pre_cts, track_ids = self._get_pre_dets(
        pre_anns, trans_input_pre, trans_output_pre)
     
      ret['pre_img'] = pre_img
      if opt.pre_hm:
        ret['pre_hm'] = pre_hm 

    # init samples
    self._init_ret(ret)
    # ret ={'image': curr_img after aug,
    #       'pre_img': pre_img after the same aug,
    #       'pre_hm': Previous frame GT heat map with network input size
    #       'hm': shape(1, h/4, w/4), 
    #       'ind':shape(K,), 'cat':shape(K,), 'mask':shape(K,), 
    #       'reg':shape(k, 2), 'reg_mask':shape(k, 2), 
    #       'tracking': shape(k, 2), 'tracking_mask':shape(k, 2),
    #       'ltrb_amodal': shape(k, 4), 'ltrb_amodal_mask':shape(k, 4)
    #       'reid':shape(k,), 'reid_mask': shape(k,)}
    
    num_objs = min(len(anns), self.max_objs)
    
    # vis 
    # im = ret['image'].copy().transpose(1,2,0)
    # im = (((np.ascontiguousarray(im)) * self.std  + self.mean) * 255).astype(np.uint8)
    
    for k in range(num_objs):
      ann = anns[k]
      # ann = { 'id': id,
      #         'category_id':cls_id,
      #         'image_id': img_id,
      #         'track_id': track_id,
      #         'bbox': x1y1wh,
      #         'conf': )} 
      cls_id = int(self.cat_ids[ann['category_id']])
      if cls_id > self.opt.num_classes or cls_id <= 0:
        continue
      bbox, bbox_amodal = self._get_bbox_output(
        ann['bbox'], trans_output, height, width)
      
      # vis
      # im = cv2.rectangle(im, ( int(bbox[0]*4), int(bbox[1]*4) ), ( int(bbox[2]*4), int(bbox[3]*4) ), color=(0, 0, 255), thickness=1)

      if cls_id <= 0 or ('iscrowd' in ann and ann['iscrowd'] > 0):
        self._mask_ignore_or_crowd(ret, cls_id, bbox) 
        continue

      self._add_instance(
        ret, k, cls_id, bbox, bbox_amodal, ann, pre_cts, track_ids)
    # cv2.imwrite('./ddd.jpg', im)
    return ret


  def get_default_calib(self, width, height):
    calib = np.array([[self.rest_focal_length, 0, width / 2, 0], 
                        [0, self.rest_focal_length, height / 2, 0], 
                        [0, 0, 1, 0]])
    return calib

  def _load_image_anns(self, img_id, coco, img_dir):
    img_info = coco.loadImgs(ids=[img_id])[0]
    file_name = img_info['file_name']
    img_path = osp.join(img_dir, file_name)
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    anns = copy.deepcopy(coco.loadAnns(ids=ann_ids))
    img = cv2.imread(img_path)
    return img, anns, img_info, img_path

  def _load_data(self, index):# idx 
    coco = self.coco
    img_dir = self.img_dir
    img_id = self.images[index]
    img, anns, img_info, img_path = self._load_image_anns(img_id, coco, img_dir)

    return img, anns, img_info, img_path


  def _load_pre_data(self, video_id, frame_id, sensor_id=1):
    img_infos = self.video_to_images[video_id]
    # If training, random sample nearby frames as the "previoud" frame
    # If testing, get the exact prevous frame
    if 'train' in self.split:
      img_ids = [(img_info['id'], img_info['frame_id']) \
          for img_info in img_infos \
          if abs(img_info['frame_id'] - frame_id) < self.opt.max_frame_dist and \
          (not ('sensor_id' in img_info) or img_info['sensor_id'] == sensor_id)]
    else:
      img_ids = [(img_info['id'], img_info['frame_id']) \
          for img_info in img_infos \
            if (img_info['frame_id'] - frame_id) == -1 and \
            (not ('sensor_id' in img_info) or img_info['sensor_id'] == sensor_id)]
      if len(img_ids) == 0:
        img_ids = [(img_info['id'], img_info['frame_id']) \
            for img_info in img_infos \
            if (img_info['frame_id'] - frame_id) == 0 and \
            (not ('sensor_id' in img_info) or img_info['sensor_id'] == sensor_id)]
    rand_id = np.random.choice(len(img_ids)) 
    img_id, pre_frame_id = img_ids[rand_id]
    frame_dist = abs(frame_id - pre_frame_id)
    img, anns, _, _ = self._load_image_anns(img_id, self.coco, self.img_dir)
    return img, anns, frame_dist

  def _get_pre_dets(self, anns, trans_input, trans_output):
    hm_h, hm_w = self.opt.input_h, self.opt.input_w 
    down_ratio = self.opt.down_ratio# 4
    trans = trans_input
    reutrn_hm = self.opt.pre_hm
    pre_hm = np.zeros((1, hm_h, hm_w), dtype=np.float32) if reutrn_hm else None
    pre_cts, track_ids = [], []
    for ann in anns:
      cls_id = int(self.cat_ids[ann['category_id']])
      if cls_id > self.opt.num_classes or cls_id <= 0 or \
         ('iscrowd' in ann and ann['iscrowd'] > 0):
        continue
      bbox = self._coco_box_to_bbox(ann['bbox'])# x1y1wh 2 xyxy
      bbox[:2] = affine_transform(bbox[:2], trans)
      bbox[2:] = affine_transform(bbox[2:], trans)
      bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, hm_w - 1)
      bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, hm_h - 1)
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
      # max_rad = 1
      if (h > 0 and w > 0):
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius)) 
        # max_rad = max(max_rad, radius)
        ct = np.array(
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)# cx cy 
        ct0 = ct.copy()# cx cy 
        conf = 1

        ct[0] = ct[0] + np.random.randn() * self.opt.hm_disturb * w
        ct[1] = ct[1] + np.random.randn() * self.opt.hm_disturb * h
        conf = 1 if np.random.random() > self.opt.lost_disturb else 0
        
        ct_int = ct.astype(np.int32)
        if conf == 0:
          pre_cts.append(ct / down_ratio)
        else:
          pre_cts.append(ct0 / down_ratio)
        track_ids.append(ann['track_id'] if 'track_id' in ann else -1)
        if reutrn_hm:
          draw_umich_gaussian(pre_hm[0], ct_int, radius, k=conf)
      
        if np.random.random() < self.opt.fp_disturb and reutrn_hm:
          ct2 = ct0.copy()
          # Hard code heatmap disturb ratio, haven't tried other numbers.
          ct2[0] = ct2[0] + np.random.randn() * 0.05 * w
          ct2[1] = ct2[1] + np.random.randn() * 0.05 * h 
          ct2_int = ct2.astype(np.int32)
          draw_umich_gaussian(pre_hm[0], ct2_int, radius, k=conf)

    return pre_hm, pre_cts, track_ids


  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i


  def _get_aug_param(self, c, s, width, height, disturb=False):
    # c = np.array([w/2, h/2])
    if (not self.opt.not_rand_crop) and not disturb:
      aug_s = np.random.choice(np.arange(0.6, 1.4, 0.1))
      w_border = self._get_border(128, width)
      h_border = self._get_border(128, height)
      c[0] = np.random.randint(low=w_border, high=width - w_border)
      c[1] = np.random.randint(low=h_border, high=height - h_border)
    else:
      sf = self.opt.scale
      cf = self.opt.shift
      if type(s) == float:
        s = [s, s]
      c[0] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
      c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
      aug_s = np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
    
    if np.random.random() < self.opt.aug_rot:
      rf = self.opt.rotate
      rot = np.clip(np.random.randn()*rf, -rf*2, rf*2)
    else:
      rot = 0
    
    return c, aug_s, rot

  def _flip_anns(self, anns, width):
    for k in range(len(anns)):
      bbox = anns[k]['bbox']
      anns[k]['bbox'] = [
        width - bbox[0] - 1 - bbox[2], bbox[1], bbox[2], bbox[3]]
    return anns


  def _get_input(self, img, trans_input):
    inp = cv2.warpAffine(img, trans_input, 
                        (self.opt.input_w, self.opt.input_h),
                        flags=cv2.INTER_LINEAR)
    
    inp = (inp.astype(np.float32) / 255.)
    if self.split == 'train' and not self.opt.no_color_aug:
      color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)
    return inp


  def _init_ret(self, ret):
    max_objs = self.max_objs * self.opt.dense_reg
    ret['hm'] = np.zeros(
      (self.opt.num_classes, self.opt.output_h, self.opt.output_w), 
      np.float32)# 
    ret['ind'] = np.zeros((max_objs), dtype=np.int64)
    ret['cat'] = np.zeros((max_objs), dtype=np.int64)
    ret['mask'] = np.zeros((max_objs), dtype=np.float32)
    # ret ={'image': curr_img after aug, 'pre_img': pre_img after the same aug  'pre_hm': Previous frame GT heat map with network input size
    #        'hm': shape(1, h/4, w/4), 'ind':shape(K,),  'cat':shape(K,),  'mask':shape(K,) }

    regression_head_dims = {'reg': 2, 'tracking': 2, 'ltrb_amodal': 4}

    for head in regression_head_dims:
      if head in self.opt.heads:# {'hm', 'reg', 'tracking','ltrb_amodal', 'reid' }
        ret[head] = np.zeros(
          (max_objs, regression_head_dims[head]), dtype=np.float32)
        ret[head + '_mask'] = np.zeros(
          (max_objs, regression_head_dims[head]), dtype=np.float32)
    
    ret['bbox_amodal'] = np.zeros((max_objs, 4), dtype=np.float32)

    if 'reid' in self.opt.heads:
      ret['reid'] = np.zeros((max_objs), dtype=np.int64)
      ret['reid_mask'] = np.zeros((max_objs), dtype=np.float32)
  # ret ={'image': curr_img after aug,
  #       'pre_img': pre_img after the same aug,
  #       'pre_hm': Previous frame GT heat map with network input size
  #       'hm': shape(1, h/4, w/4), 
  #       'ind':shape(K,), 'cat':shape(K,), 'mask':shape(K,), 
  #       'reg':shape(k, 2), 'reg_mask':shape(k, 2), 
  #       'tracking': shape(k, 2), 'tracking_mask':shape(k, 2),
  #       'ltrb_amodal': shape(k, 4), 'ltrb_amodal_mask':shape(k, 4)
  #       'reid': shape(k,), 'reid_mask': shape(k,)}

  def _get_calib(self, img_info, width, height):
    if 'calib' in img_info:
      calib = np.array(img_info['calib'], dtype=np.float32)
    else:
      calib = np.array([[self.rest_focal_length, 0, width / 2, 0], 
                        [0, self.rest_focal_length, height / 2, 0], 
                        [0, 0, 1, 0]])
    return calib
 

  def _ignore_region(self, region, ignore_val=1):
    np.maximum(region, ignore_val, out=region)

  def _mask_ignore_or_crowd(self, ret, cls_id, bbox):
    # mask out crowd region, only rectangular mask is supported
    # ret ={'image': img after aug,
    #       'hm': shape(1, h/4, w/4), 
    #       'ind':k, 'cat':k, 'mask':k 
    #       'reg':shape(k, 2), 'reg_mask':shape(k, 2), 
    #       'tracking': shape(k, 2), 'tracking_mask':shape(k, 2),
    #      ... }

    if cls_id == 0: # ignore all classes
      self._ignore_region(ret['hm'][:, int(bbox[1]): int(bbox[3]) + 1, 
                                        int(bbox[0]): int(bbox[2]) + 1])
    else:
      # mask out one specific class
      self._ignore_region(ret['hm'][abs(cls_id) - 1, 
                                    int(bbox[1]): int(bbox[3]) + 1, 
                                    int(bbox[0]): int(bbox[2]) + 1])

  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox# x1y1wh to xyxy

  def _get_bbox_output(self, bbox, trans_output, height, width):

    bbox = self._coco_box_to_bbox(bbox).copy()

    rect = np.array([[bbox[0], bbox[1]], [bbox[0], bbox[3]],
                    [bbox[2], bbox[3]], [bbox[2], bbox[1]]], dtype=np.float32)
    for t in range(4):
      rect[t] =  affine_transform(rect[t], trans_output)
    
    bbox[:2] = rect[:, 0].min(), rect[:, 1].min()# x1 y1
    bbox[2:] = rect[:, 0].max(), rect[:, 1].max()# x2 y2 

    bbox_amodal = copy.deepcopy(bbox)# xyxy bbox_amodal --- 
    bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.opt.output_w - 1)
    bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.opt.output_h - 1)
    return bbox, bbox_amodal

  def _add_instance(
    self, ret, k, cls_id, bbox, bbox_amodal, ann, pre_cts=None, track_ids=None):
    # ret ={'image': img after aug, 
    #       'pre_img': pre_img after the same aug,
    #       'pre_hm': Previous frame GT heat map with network input size 'hm': shape(1, h/4, w/4), 
    #       'hm': shape(1, h/4, w/4),  'ind':k,   'cat':k,   'mask':k 
    #       'reg':shape(k, 2), 'reg_mask':shape(k, 2), 
    #       'tracking': shape(k, 2), 'tracking_mask':shape(k, 2),
    #       'ltrb_amodal': shape(k, 4), 'ltrb_amodal_mask':shape(k, 4)
    #       'reid': shape(k, ), 'reid_mask': shape(k,)}
  
    h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
    if h <= 0 or w <= 0:
      return
    radius = gaussian_radius((math.ceil(h), math.ceil(w)))
    radius = max(0, int(radius)) 
    ct = np.array(
      [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
    ct_int = ct.astype(np.int32)
    ret['cat'][k] = cls_id - 1
    ret['mask'][k] = 1
    ret['ind'][k] = ct_int[1] * self.opt.output_w + ct_int[0]
    ret['reg'][k] = ct - ct_int
    ret['reg_mask'][k] = 1
    draw_umich_gaussian(ret['hm'][cls_id - 1], ct_int, radius)

    if 'tracking' in self.opt.heads:
      if ann['track_id'] in track_ids:
        pre_ct = pre_cts[track_ids.index(ann['track_id'])]
        ret['tracking_mask'][k] = 1 
        ret['tracking'][k] = pre_ct - ct_int 
  

    if 'ltrb_amodal' in self.opt.heads:
      ret['ltrb_amodal'][k] = \
        bbox_amodal[0] - ct_int[0], bbox_amodal[1] - ct_int[1], \
        bbox_amodal[2] - ct_int[0], bbox_amodal[3] - ct_int[1]
      ret['ltrb_amodal_mask'][k] = 1
      ret['bbox_amodal'][k] = (bbox_amodal)

    if 'reid' in self.opt.heads:
      ret['reid'][k]=ann['track_id']
      ret['reid_mask'][k] = 1 

  def fake_video_data(self):
    self.coco.dataset['videos'] = []
    for i in range(len(self.coco.dataset['images'])):
      img_id = self.coco.dataset['images'][i]['id']
      self.coco.dataset['images'][i]['video_id'] = img_id
      self.coco.dataset['images'][i]['frame_id'] = 1
      self.coco.dataset['videos'].append({'id': img_id})
    
    if not ('annotations' in self.coco.dataset):
      return

    for i in range(len(self.coco.dataset['annotations'])):
      self.coco.dataset['annotations'][i]['track_id'] = i + 1
