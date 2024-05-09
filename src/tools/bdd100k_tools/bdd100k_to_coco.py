# train for centertrack 
import cv2
import os
import json
import tqdm
import numpy as np

labels_path = '/data2/project/BDD100K-MOT/labels'
img_path = '/data2/project/BDD100K-MOT/images'

out_path = '/data2/project/BDD100K-MOT/'

split = ['train','val']
categories = [
    {"id": 1, "name": "pedestrian"},
    {"id": 2, "name": "rider"},
    {"id": 3, "name": "car"},
    {"id": 4, "name": "truck"},
    {"id": 5, "name": "bus"},
    {"id": 6, "name": "train"},
    {"id": 7, "name": "motorcycle"},
    {"id": 8, "name": "bicycle"},
]
cat = {"pedestrian":1, "rider":2, "car":3, "truck":4, "bus":5, "train":6, "motorcycle":7, "bicycle":8}
# 1: pedestrian
# 2: rider
# 3: car
# 4: truck
# 5: bus
# 6: train
# 7: motorcycle
# 8: bicycle  
# 9: traffic light --- Don't need tracking
# 10: traffic sign  ---   Don't need tracking
# For MOT and MOTS, only the first 8 classes are used and evaluated. 
# But for generating data json files, all of classes should be recorded

for s in split:
    img_id = 1; ann_id = 1; video_cnt = 0
    images = []; annotations=[]; videos = []
    video_labels_list = [d for d in os.listdir(os.path.join(labels_path, s)) if '.json' in d]
    
    for v_label in tqdm.tqdm(video_labels_list):
        video_cnt += 1
        video = {'id': video_cnt, 'file_name':v_label[:-5]}
        videos.append(video)
        
        v_lab_path = os.path.join(os.path.join(labels_path, s, v_label))
        with open(v_lab_path, 'r') as f:
            annos=json.load(f)# anns per video
        num_frames  = len(annos)# the number of frames per video
        for ann in annos:# ann --- per img, imcluding frames without ann
            
            img_name = os.path.join(img_path, s, ann['videoName'], ann['name'])
            img=cv2.imread(img_name)
            h,w,_ = img.shape
            
            img_info = {
            'file_name':img_name,
            'width':w,
            'height':h,
            'id': img_id,
            'frame_id': ann['frameIndex'] + 1,
            'prev_image_id': -1 if ann['frameIndex'] == 0 else img_id - 1,
            'next_image_id': -1 if ann['frameIndex'] == num_frames-1 else img_id + 1,
            'video_id': video_cnt
            }
            images.append(img_info)
            
            for lab in ann['labels']:
                #  lab---per object  
                if lab['category'] in cat:
                    pass
                else:
                    continue
                    
                track_id = lab['id']    

                is_crowd = lab['attributes']['crowd']
                x1, y1, x2, y2=lab['box2d']['x1'], lab['box2d']['y1'], lab['box2d']['x2'], lab['box2d']['y2']
                
                annotation = {
                    'image_id': img_id,
                    'score': 1,
                    'bbox': [x1, y1, x2-x1, y2-y1],
                    'category_id': cat[lab['category']],
                    'id': ann_id,
                    'iscrowd':  1 if is_crowd else 0,
                    'track_id': int(track_id),
                    'segmentation': [],
                    'area': (x2-x1)*(y2-y1)   
                }
                annotations.append(annotation)
                ann_id += 1
                    
            img_id += 1
            
    dataset_dict = {}
    dataset_dict["images"] = images
    dataset_dict["annotations"] = annotations
    dataset_dict["categories"] = categories
    dataset_dict["videos"] = videos
    
    json_str = json.dumps(dataset_dict)
    print(f' The number of detection objects is {ann_id - 1}, The number of detection imgs is {img_id -1} .')
    with open(out_path+f'{s}.json', 'w') as json_file:
        json_file.write(json_str)
            
       
        

