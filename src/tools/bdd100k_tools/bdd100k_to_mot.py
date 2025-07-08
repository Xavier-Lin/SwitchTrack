
import tqdm
import cv2
import os
import json
labels_path = '/data/zelinliu/BDD100K-MOT/labels'
out_path = '/data/zelinliu/BDD100K-MOT/GT/'
split = ['train', 'val']


cat = {"pedestrian":1, "rider":2, "car":3, "truck":4, "bus":5, "train":6, "motorcycle":7, "bicycle":8}

save_format = '{frame},{id},{x1},{y1},{w},{h},{s},{cls},1\n'

for s in split: # train  val 
    videos_labels_path = os.path.join(labels_path, s)
    v_anns_list = [i for i in os.listdir(videos_labels_path) if '.json' in i]
    out_dir = out_path + s
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    for v_anns in tqdm.tqdm(v_anns_list):
        frame_id = 0 

        v_anns_path = os.path.join(videos_labels_path, v_anns)
        with open(v_anns_path, 'r') as f:
            v_labels=json.load(f) 
        
        filename = os.path.join(out_dir, v_anns[:-5] + '.txt')  
        
        with open(filename, 'w') as a:
            for i, img_anns in enumerate(v_labels):
                if len(img_anns['labels']) ==0 :
                    continue
               
                frameidx = img_anns['frameIndex']
                frame_id = frameidx + 1 
                
                for lab in img_anns['labels']:
                    track_id = lab['id']   
                                    
                    x1 = lab['box2d']['x1']
                    y1 = lab['box2d']['y1']
                    w = lab['box2d']['x2'] - lab['box2d']['x1'] 
                    h = lab['box2d']['y2'] - lab['box2d']['y1'] 
                    
                    score = 1 if lab['category'] in cat else 0
                    cls_id = cat[lab['category']] if lab['category'] in cat else -1
                    line = save_format.format(frame=int(frame_id), id=int(track_id), x1=x1, y1=y1, w=w, h=h, s=score, cls=cls_id)
                    a.write(line)
                
                
                
             
    

