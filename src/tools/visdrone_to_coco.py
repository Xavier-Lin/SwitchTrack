# from multiprocessing.util import is_exiting
import os
import numpy as np
import tqdm
import json

DATA_PATH = '/data/zelinliu/visdrone_mot/'
OUT_PATH = '/data/zelinliu/visdrone_mot/labels_with_ids/'
SPLITS = ['test',]
cat = {0:'ignored_regions', 1:'pedestrian',
       2:'people', 3:'bicycle', 4:'car', 5:'van', 6:'truck', 
       7:'tricycle', 8:'awning-tricycle', 9:'bus', 10:'motor', 11:'others'}
# For mot only needs pedestrian car van truck bus -- 1 4 5 6 9
gt_cat = {'pedestrian':1, 'people':2, 'bicycle':3, 'car':4, 'van':5, 
          'truck':6, 'tricycle':7,'awning-tricycle':8, 'bus':9,
          'motor':10}
def mikdir(ph):
  if not os.path.exists(ph):
    os.makedirs(ph)

if __name__ == '__main__':
  for split in SPLITS:
    data_path = DATA_PATH + split# /data/zelinliu/visdrone_mot/train
    mikdir(OUT_PATH)
    out_path = OUT_PATH + '{}.json'.format(split)# /data/zelinliu/visdrone_mot/labels_with_ids/train.json
    out = {'images': [], 'annotations': [], 
           'categories': [{'id': 1, 'name': 'vehicle'}],
           'videos': []}
    seqs = os.listdir(os.path.join(data_path, 'sequences'))
    seqs = [s for s in seqs if not ".D" in s]
    image_cnt = 0
    ann_cnt = 0
    video_cnt = 0
    tid_curr = 0
    tid_last = -1

    for seq in tqdm.tqdm(sorted(seqs)):
      if '.DS_Store' in seq:
        continue
      
      video_cnt += 1
      out['videos'].append({
        'id': video_cnt,
        'file_name': seq})
      
      seq_path = '{}/{}/{}'.format(data_path, 'sequences', seq)
      ann_path = os.path.join(data_path, 'annotations') + f'/{seq}.txt'
      images = os.listdir(seq_path)
      num_images = len([image for image in images if 'jpg' in image])
      anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
      
      image_range = [0, num_images - 1]

      for i in range(num_images): 
        image_info = {'file_name': '{}/{:07d}.jpg'.format(seq, i + 1),
                      'id': image_cnt + i + 1,
                      'frame_id': i + 1 - image_range[0],
                      'prev_image_id': image_cnt + i if i > 0 else -1,
                      'next_image_id': \
                        image_cnt + i + 2 if i < num_images - 1 else -1,
                      'video_id': video_cnt}
        out['images'].append(image_info)
        

      for i in range(anns.shape[0]):
        if int(anns[i][6]) == 0:
          continue
        elif int(anns[i][7]) in {0, 11}:
          continue
        else:
          category_id = gt_cat[cat[int(anns[i][7])]]
        frame_id = int(anns[i][0])
        tid = int(anns[i][1])
        if not tid == tid_last:
          tid_curr += 1
          tid_last = tid
        ann_cnt += 1
        ann = { 'id': ann_cnt,
                'category_id': category_id,
                'image_id': image_cnt + frame_id,
                'track_id': tid_curr,
                'bbox': anns[i][2:6].tolist(),
                'conf': float(anns[i][6])}
        out['annotations'].append(ann)
      image_cnt += num_images

    print('loaded {} for {} images and {} samples'.format(
      split, len(out['images']), len(out['annotations'])))
    json.dump(out, open(out_path, 'w'))