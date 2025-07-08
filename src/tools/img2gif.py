import os
from PIL import Image


def imgs2gif(imgPaths, saveName, duration=None, loop=0, fps=None):
    """
    Generate gif 
    :param imgPaths:the path of a wqxvideo seq
    :param saveName: the name of gif
    :param duration: the time per frame(s)
    :param fps: fps
    :param loop: show times
    :return:
    """
    if fps:
        duration = 1 / fps
    duration *= 1000
    imgs = [Image.open(str(path)) for path in imgPaths]
    imgs[0].save(saveName, save_all=True, append_images=imgs, duration=duration, loop=loop)

data_path = 'results'
pathlist = [s for s in os.listdir(data_path) if 'demo' in s]
img_path={}
for i,dir in enumerate(pathlist) :
    d=dir.replace('demo','')
    # d='{:06d}.jpg'.format(int(d[:-4]))
    img_path[int(d[:-4])]=dir
img_key = sorted(img_path)
gifkey = img_key[640:701]
p_lis = []
for n, p in enumerate(gifkey):
    p_lis.append(os.path.join(data_path,img_path[p]))

imgs2gif(p_lis, "moiton.gif", 0.5, 1)


