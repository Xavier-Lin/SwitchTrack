from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import cv2
from .image import transform_preds_with_trans, get_affine_transform


def get_alpha(rot):
  idx = rot[:, 1] > rot[:, 5]
  alpha1 = np.arctan2(rot[:, 2], rot[:, 3]) + (-0.5 * np.pi)
  alpha2 = np.arctan2(rot[:, 6], rot[:, 7]) + ( 0.5 * np.pi)
  return alpha1 * idx + alpha2 * (1 - idx)

def generic_post_process(
  opt, dets, c, s, h, w, num_classes, calibs=None, height=-1, width=-1):
  if not ('scores' in dets):
    return [{}], [{}]
  ret = []
  for i in range(len(dets['scores'])):
    preds = []
    trans = get_affine_transform(
      c[i], s[i], 0, (w, h), inv=1).astype(np.float32)
    for j in range(len(dets['scores'][i])):
      if dets['scores'][i][j] < opt.track_thresh:
        break
     
      item = {}
      item['score'] = dets['scores'][i][j]
      item['class'] = int(dets['clses'][i][j]) + 1  
      item['ct'] = transform_preds_with_trans(
        (dets['ct_acc'][i][j]).reshape(1, 2), trans).reshape(2)

      if 'tracking' in dets:
        tracking = transform_preds_with_trans(
          (dets['tracking'][i][j] + dets['cts'][i][j]).reshape(1, 2), 
          trans).reshape(2)
         
        item['tracking'] = tracking - item['ct'] 

      if 'bboxes' in dets:
        bbox = transform_preds_with_trans(
          dets['bboxes'][i][j].reshape(2, 2), trans).reshape(4)
        
        bbox[0:4:2] = np.clip(bbox[0:4:2], 0, width-1)
        bbox[1:4:2] = np.clip(bbox[1:4:2], 0, height-1)
        item['bbox'] = bbox
        
      if 'reid_feature' in dets:
        item['reid'] = dets['reid_feature'][i][j]
      
      preds.append(item)
   
    # if 'velocity' in dets:
    #   for j in range(len(preds)):
    #     preds[j]['velocity'] = dets['velocity'][i][j]

    ret.append(preds)
  
  return ret