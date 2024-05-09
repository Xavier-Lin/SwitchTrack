import tqdm
import os
import os.path as osp
import numpy as np
import cv2

labels_path = '/data/zelinliu/BDD100K-MOT/GT/train/00a9cd6b-b39be004.txt'
data_path = '/data/zelinliu/BDD100K-MOT/images/train/00a9cd6b-b39be004'
out_path = './test_label/'

if not osp.exists(out_path):
  os.makedirs(out_path)
imgs_list = [ i for i in os.listdir(data_path) if '.jpg' in i]
imgs_list = sorted(imgs_list)

for n, img_n in enumerate(imgs_list):
  img_dir = osp.join(data_path, img_n)
  im = cv2.imread(img_dir)
  anns = np.loadtxt(labels_path, dtype=np.float32, delimiter=',')
  
  frame_label = []
  t_ids = []
  
  for ann in anns:
    if ann[6] == 0:
      continue
    if ann[0] == n+1:
      x1,y1,w,h = ann[2:6]
      x2 = x1 + w
      y2 = y1 + h
      bbox = [x1,y1,x2,y2]
      frame_label.append(bbox)
      t_ids.append(ann[1])
    else:
      continue
    
  for bbox, tid in zip(frame_label,t_ids) :
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = '{}'.format(int(tid))
    txt_size = cv2.getTextSize(text, font, 0.8, 1)[0]
    cv2.rectangle(
      im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), 
      color=(0,0,255), thickness=1, lineType=cv2.LINE_AA)
    cv2.putText(im, text, (int(bbox[0]), int(bbox[1])+txt_size[1]), font, 0.8, color=(0, 255, 0), thickness=1)
    
  cv2.imwrite(f'{out_path}'+f'{n}'+'.jpg', im)
    

