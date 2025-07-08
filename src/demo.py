from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import sys
import cv2
import torch
import numpy as np
from lib.opts import opts
from lib.detector import Detector

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge', 'display']

def demo(opt):
  torch.cuda.set_device(opt.test_device)
  opt.debug = max(opt.debug, 1)

  detector = Detector(opt)
  detector.reset_tracking()
  
  is_video = False
  # Demo on images sequences
  if os.path.isdir(opt.demo):
    image_names = []
    ls = os.listdir(opt.demo)
    for file_name in sorted(ls):
        ext = file_name[file_name.rfind('.') + 1:].lower()
        if ext in image_ext:
            image_names.append(os.path.join(opt.demo, file_name))
  else:
    image_names = [opt.demo]

  # Initialize output video
  
  out_name = opt.demo[opt.demo.rfind('/') + 1:]
  print('out_name', out_name)
  os.makedirs('results',exist_ok= True)
  if opt.save_video:
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter('./results/{}.avi'.format(
      opt.exp_id + '_' + out_name),fourcc, opt.save_framerate, (
        opt.video_w, opt.video_h))
  
  if opt.debug < 5:
    detector.pause = False
  cnt = 0
  results = {}
  
  while True:
      if cnt < len(image_names):
        img = cv2.imread(image_names[cnt])
      else:
        save_and_exit(opt, out, results, out_name)
      cnt += 1

      img = cv2.resize(img, (opt.video_w, opt.video_h))
      ret = detector.run(img)

      # results[cnt] is a list of dicts:
      #  [{'bbox': [x1, y1, x2, y2], 'tracking_id': id, 'category_id': c, ...}]
      results[cnt] = ret['results']

      # save debug image to video
      if opt.save_video:
        out.write(ret['generic'])
        if not is_video:
          cv2.imwrite('./results/demo{}.jpg'.format(cnt), ret['generic'])
      
      # esc to quit and finish saving video
      if cnt == len(image_names):
        # import pdb
        # pdb.set_trace()
        save_and_exit(opt, out, results, out_name)
        break


def save_and_exit(opt, out=None, results=None, out_name=''):
  if opt.save_video and out is not None:
    out.release()
  sys.exit(0)

def _to_list(results):
  for img_id in results:
    for t in range(len(results[img_id])):
      for k in results[img_id][t]:
        if isinstance(results[img_id][t][k], (np.ndarray, np.float32)):
          results[img_id][t][k] = results[img_id][t][k].tolist()
  return results

if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
  