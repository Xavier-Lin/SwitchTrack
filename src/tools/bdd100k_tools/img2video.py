import glob as gb
import cv2
import tqdm
import os

def img2video(visual_path="visual_val_gt"):
    print("Starting img2video")
    img_paths = gb.glob(visual_path + "/*.jpg") 

    img_name={}
    print("sort demo imgs")
    for name in tqdm.tqdm(img_paths):
        na=int(os.path.basename(name)[:-4])
        im=cv2.imread(name)
        img_name[na]=im

    fps = 5
    size = ((cv2.imread(img_paths[0])).shape[1], (cv2.imread(img_paths[0])).shape[0])
    videowriter = cv2.VideoWriter(visual_path +'/../'+ "_video.avi",cv2.VideoWriter_fourcc('M','J','P','G'), fps, size)
    
    print("save imgs to videos")
    imgn=sorted(img_name)
    for i in tqdm.tqdm(imgn):
        img = cv2.resize(img_name[i], size)
        videowriter.write(img)

    videowriter.release()
    print("img2video Done")

img2video('./test_label')