from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import copy
import numpy as np
import torch
import math
from loguru import logger

from model.model import create_model, load_model
from model.decode import generic_decode
from model.utils import fuse_model

from utils.image import get_affine_transform, affine_transform
from utils.image import draw_umich_gaussian, gaussian_radius
from utils.post_process import generic_post_process
from utils.debugger import Debugger
from dataset.dataset_factory import get_dataset

from tracker_zoo.dctrack import Tracker
from tracker_zoo.fairmot_tracker import JDETracker
from tracker_zoo.sort_tracker import Sort
from tracker_zoo.center_tracker import CTracker 
from tracker_zoo.center_tracker_kf import KFCTracker
from tracker_zoo.deepsort_tracker.deepsort import DeepSort
from tracker_zoo.bytetrack import BYTETracker
from tracker_zoo.ocsort import OCSort


class Detector(object):
  def __init__(self, opt):
    if torch.cuda.is_available():
      opt.device = torch.device('cuda', opt.test_device)
    else:
      opt.device = torch.device('cpu')
    logger.info('Creating model...')
    self.model = create_model(
      opt.arch, opt.heads, opt.head_conv, opt=opt)# get DLA34
    self.model = load_model(self.model, opt.load_model, opt)
    self.model = self.model.to(opt.device)
    self.model.eval()
    
    if opt.fuse:
      logger.info("\tFusing model...")
      self.model = fuse_model(self.model)
    
    self.opt = opt
    self.trained_dataset = get_dataset(opt.dataset)# custom
    self.mean = np.array(
      self.trained_dataset.mean, dtype=np.float32).reshape(1, 1, 3)
    self.std = np.array(
      self.trained_dataset.std, dtype=np.float32).reshape(1, 1, 3)
    self.rest_focal_length = self.trained_dataset.rest_focal_length 
     
    self.flip_idx = self.trained_dataset.flip_idx#[[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
    self.cnt = 0
    self.pre_images = None
    self.pre_image_ori = None
    if opt.use_fairmot:
      self.tracker = JDETracker(opt)
    elif opt.use_sort:
      self.tracker = Sort(det_thresh=opt.track_thresh)
    elif opt.use_ocsort:
      self.tracker = OCSort(det_thresh=opt.track_thresh)
    elif opt.use_center:
      self.tracker = CTracker(opt)
    elif opt.use_center_kf:
      self.tracker = KFCTracker(opt)
    elif opt.use_deepsort:
      self.tracker = DeepSort()
    elif opt.use_byte:
      self.tracker = BYTETracker(opt)
    else:
      self.tracker = Tracker(opt)
    self.debugger = Debugger(opt=opt, dataset=self.trained_dataset)


  def run(self, image_or_path_or_tensor, meta={}):
    '''start tracking '''
    # init debugger
    if self.opt.debug >= 0:
      self.debugger.clear()
      image = image_or_path_or_tensor
      pre_processed = False
    else:
      image = image_or_path_or_tensor['image'][0].numpy() # inference process--- original img of current frame
      pre_processed_images = image_or_path_or_tensor
      pre_processed = True
    
    '''begain to run pre_process and main_process and post_process''' 
    detections = []# current dets

    for scale in self.opt.test_scales:# opt.test_scales -- an equal scaling factor for image size during testing
      ''' run pre_process '''
      if not pre_processed:
        # no_prefetch testing or demo
        images, meta = self.pre_process(image, scale, meta)
      else:
        # prefetch testing
        images = pre_processed_images['images'][scale][0] 
        meta = pre_processed_images['meta'][scale] 
        meta = {k: v.numpy()[0] for k, v in meta.items()} 
        if 'pre_dets' in pre_processed_images['meta']:
          meta['pre_dets'] = pre_processed_images['meta']['pre_dets']
        if 'cur_dets' in pre_processed_images['meta']:
          meta['cur_dets'] = pre_processed_images['meta']['cur_dets']
      
      images = images.to(self.opt.device, non_blocking=self.opt.non_block_test)

      # initializing tracker
      pre_hms, pre_inds = None, None 
      if self.opt.tracking:# for demo and test :T
        if self.pre_images is None:
          self.pre_images = images 

        if self.opt.pre_hm:#t
          # render input heatmap from tracker statues which contained object information of previous frame 
          pre_hms, pre_inds = self._get_additional_inputs(
            self.tracker.tracks, meta, with_hm=not self.opt.zero_pre_hm)
      
      '''run process: 
          network forward and decode the output of network -- the output feature maps, only used for visualizing''' 
      output, dets = self.process(
        images, self.pre_images, pre_hms, pre_inds)# output, dets
        # dets :{'scores': 'clses':  'xs':  'ys': 'cts': 'ct_acc': 'bboxes_amodal': 'bboxes'
        #       'tracking':  'pre_cts':   'reid_feature':   }
      '''run post_process: 将所有输出尺寸下的 目标信息 转换为 输入尺寸下的目标信息'''
      # back to the input image coordinate system
      result = self.post_process(dets, meta, scale)
      #result =  [{'score':
      #            'class':
      #            'ct':
      #            'tracking':
      #            'bbox':
      #            'reid_feature': },...]

      # for x in result: # ---- only used for vis
      #   image[int(x['ct'][1])-2:int(x['ct'][1])+3, int(x['ct'][0])-2:int(x['ct'][0])+3, : ] = np.array([255,255,255],dtype=np.uint8)
      #     image[int( (x['bbox'][1]+x['bbox'][3]) /2 - 2):int( (x['bbox'][1]+x['bbox'][3]) /2 + 3), \
      #               int((x['bbox'][0]+x['bbox'][2]) /2 - 2):int((x['bbox'][0]+x['bbox'][2]) /2 + 3),: ] = np.array([0,0,255],dtype=np.uint8)
      #   cv2.rectangle(image,(int(x['bbox'][0]),int(x['bbox'][1])),(int(x['bbox'][2]),int(x['bbox'][3])),color=(200,200,200),thickness=2,lineType=cv2.LINE_AA)
      # imwrite('./sss.jpg',image)

      detections.append(result)

      ''' run debugger : the self.debug only used for visualizing '''
      if self.opt.debug >= 2:
        self.debug(
          self.debugger, images, result, output, scale, 
          pre_images=self.pre_images,
          pre_hms=pre_hms)
        #self.debugger.imgs['predict_hm']、 self.debugger.imgs['pre_img']、self.debugger.imgs['pre_hm']
        #used for visualizing 

    ''' run tracking '''
    if self.opt.tracking:# T 
      if not self.opt.use_deepsort:
        results = self.tracker.step(detections[0])  
      else:
        results = self.tracker.step(detections[0], image)
     
      self.pre_images = images
     
      # only used for visualizing
    if self.opt.debug >= 1:
      self.show_results(self.debugger, image, results)
      
    self.cnt += 1 
    
    ''' return results '''
    ret = {'results': results}
    if self.opt.save_video:
      try:
        ret.update({'generic': self.debugger.imgs['generic']})
      except:
        pass
    return ret


  def _transform_scale(self, image, scale=1):
    '''
      Prepare input image in different testing modes.
        Currently support: fix short size/ center crop to a fixed size/ 
        keep original resolution but pad to a multiplication of 32
    '''
    height, width = image.shape[0:2]
    new_height = int(height * scale)
    new_width  = int(width * scale)
    if self.opt.fix_short > 0:
      if height < width:
        inp_height = self.opt.fix_short
        inp_width = (int(width / height * self.opt.fix_short) + 63) // 64 * 64
      else:
        inp_height = (int(height / width * self.opt.fix_short) + 63) // 64 * 64
        inp_width = self.opt.fix_short
      c = np.array([width / 2, height / 2], dtype=np.float32)
      s = np.array([width, height], dtype=np.float32)
    elif self.opt.fix_res:
      inp_height, inp_width = self.opt.input_h, self.opt.input_w
      c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
      s = max(height, width) * 1.0 
      # s = np.array([inp_width, inp_height], dtype=np.float32)
    else:
      inp_height = (new_height | self.opt.pad) + 1
      inp_width = (new_width | self.opt.pad) + 1
      c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
      s = np.array([inp_width, inp_height], dtype=np.float32)
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image, c, s, inp_width, inp_height, height, width

  def pre_process(self, image, scale, input_meta={}):
    '''Crop, resize, and normalize image. Gather meta data for post processing and tracking.'''
    resized_image, c, s, inp_width, inp_height, height, width = \
      self._transform_scale(image)
    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    out_height =  inp_height // self.opt.down_ratio
    out_width =  inp_width // self.opt.down_ratio 
    trans_output = get_affine_transform(c, s, 0, [out_width, out_height])

    inp_image = cv2.warpAffine(
      resized_image, trans_input, (inp_width, inp_height),
      flags=cv2.INTER_LINEAR)
    inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
    images = torch.from_numpy(images)
    meta = {'calib': np.array(input_meta['calib'], dtype=np.float32) \
             if 'calib' in input_meta else \
             self._get_default_calib(width, height)}# calib 
    meta.update({'c': c, 's': s, 'height': height, 'width': width,
            'out_height': out_height, 'out_width': out_width,
            'inp_height': inp_height, 'inp_width': inp_width,
            'trans_input': trans_input, 'trans_output': trans_output})
    
    if 'pre_dets' in input_meta:
      meta['pre_dets'] = input_meta['pre_dets']
    if 'cur_dets' in input_meta:
      meta['cur_dets'] = input_meta['cur_dets']
    return images, meta


  def _trans_bbox(self, bbox, trans, width, height):
    '''
    将上一帧 符合置信度要求的 跟踪的目标框坐标 变换为 输入尺寸下的框坐标
    '''
    bbox = np.array(copy.deepcopy(bbox), dtype=np.float32)
    bbox[:2] = affine_transform(bbox[:2], trans)
    bbox[2:] = affine_transform(bbox[2:], trans)
    bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, width - 1)
    bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, height - 1)
    return bbox

  def _get_additional_inputs(self, dets, meta, with_hm=True):
    '''Render input heatmap from previous trackings.
    '''
    trans_input, trans_output = meta['trans_input'], meta['trans_output']
    inp_width, inp_height = meta['inp_width'], meta['inp_height']
    out_width, out_height = meta['out_width'], meta['out_height']
    input_hm = np.zeros((1, inp_height, inp_width), dtype=np.float32)
   
    output_inds = []
    for det in dets:# dets 
      if (
          (
            det.score if not (self.opt.use_sort or self.opt.use_center or self.opt.use_ocsort) else det['score']
          ) < self.opt.pre_thresh
        ) or (
                False if not self.opt.use_center else det['active'] == 0
             ): 
  
        continue
      bbox = self._trans_bbox(det.tlbr if not (self.opt.use_sort or self.opt.use_center or self.opt.use_ocsort) else det['bbox'], trans_input, inp_width, inp_height)
 
      bbox_out = self._trans_bbox(
        det.tlbr if not (self.opt.use_sort or self.opt.use_center or self.opt.use_ocsort) else det['bbox'], trans_output, out_width, out_height)
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
      if (h > 0 and w > 0):
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        ct = np.array(
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        if with_hm:#T
          draw_umich_gaussian(input_hm[0], ct_int, radius)

        ct_out = np.array(
          [(bbox_out[0] + bbox_out[2]) / 2, 
           (bbox_out[1] + bbox_out[3]) / 2], dtype=np.int32)
        output_inds.append(ct_out[1] * out_width + ct_out[0])

    if with_hm:#T
      input_hm = input_hm[np.newaxis]
      input_hm = torch.from_numpy(input_hm).to(self.opt.device)# np -> tensor

    output_inds = np.array(output_inds, np.int64).reshape(1, -1)
    output_inds = torch.from_numpy(output_inds).to(self.opt.device)
    return input_hm, output_inds

  def _get_default_calib(self, width, height):
    calib = np.array([[self.rest_focal_length, 0, width / 2, 0], 
                        [0, self.rest_focal_length, height / 2, 0], 
                        [0, 0, 1, 0]])
    return calib


  def _sigmoid_output(self, output):# [{'hm':(b,1,h/4,w/4),'reg':(b,2,h/4,w/4),'wh':(b,2,h/4,w/4)},'tracking':(b,2,h/4,w/4)]
    if 'hm' in output:
      output['hm'] = output['hm'].sigmoid_()
    return output 

  def process(self, images, pre_images=None, pre_hms=None, pre_inds=None):
    with torch.no_grad():
      output = self.model(images, pre_images, pre_hms)[-1]
      # output = [{'hm':(b,1,h/4,w/4),'reg':(b,2,h/4,w/4),'ltrb_amodal':(b,4,h/4,w/4)},'tracking':(b,2,h/4,w/4)]
      output = self._sigmoid_output(output)
      output.update({'pre_inds': pre_inds})
      # output = [{'hm':(b,1,h/4,w/4),'reg':(b,2,h/4,w/4),'ltrb_amodal':(b,4,h/4,w/4)},'tracking':(b,2,h/4,w/4), 'pre_inds':()]
      dets = generic_decode(output, K=self.opt.K, opt=self.opt)
      for k in dets:
        dets[k] = dets[k].detach().cpu().numpy()
    
    return output, dets
  

  def post_process(self, dets, meta, scale=1):
    dets = generic_post_process(
      self.opt, dets, [meta['c']], [meta['s']],
      meta['out_height'], meta['out_width'], self.opt.num_classes,
      [meta['calib']], meta['height'], meta['width']
    )
    self.this_calib = meta['calib']
    
    if scale != 1:
      for i in range(len(dets[0])):
        for k in ['bbox', 'hps']:
          if k in dets[0][i]:
            dets[0][i][k] = (np.array(
              dets[0][i][k], np.float32) / scale).tolist()
    return dets[0]

  def debug(self, debugger, images, dets, output, scale=1, 
    pre_images=None, pre_hms=None):
    img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(((
      img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)
    pred = debugger.gen_colormap(output['hm'][0].detach().cpu().numpy())
    debugger.add_blend_img(img, pred, 'pred_hm')
    if pre_images is not None:
      pre_img = pre_images[0].detach().cpu().numpy().transpose(1, 2, 0)
      pre_img = np.clip(((
        pre_img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)
      debugger.add_img(pre_img, 'pre_img')
      if pre_hms is not None:
        pre_hm = debugger.gen_colormap(
          pre_hms[0].detach().cpu().numpy())
        debugger.add_blend_img(pre_img, pre_hm, 'pre_hm')

  def show_results(self, debugger, image, results):
    #current frame (original image)  tracking results of current frame
    debugger.add_img(image, img_id='generic')
    # import pdb 
    # pdb.set_trace()
    if self.opt.tracking:
      debugger.add_img(self.pre_image_ori if self.pre_image_ori is not None else image, 
        img_id='previous')
      self.pre_image_ori = image
    
    for j in range(len(results)):
      item = results[j]
      if not (self.opt.use_sort or self.opt.use_deepsort or self.opt.use_ocsort or self.opt.use_center):
        sc = item.score if self.opt.demo == '' else item.track_id
        sc = item.track_id if self.opt.show_track_color else sc # track -id
      
      if self.opt.use_fairmot or self.opt.use_center_kf or self.opt.use_byte:
        debugger.add_coco_bbox(
          item.tlbr, 0, sc, img_id='generic')
      elif self.opt.use_sort or self.opt.use_deepsort or self.opt.use_ocsort:
        debugger.add_coco_bbox(
          item[:4], 0, item[4], img_id='generic')  
      elif self.opt.use_center:
        sc = item['score'] if self.opt.demo == '' else  item['tracking_id']
        debugger.add_coco_bbox(
          item['bbox'], 0, sc, img_id='generic')
        debugger.add_arrow(item['ct'], item['tracking'], img_id='generic')
      else:
        debugger.add_coco_bbox(
          item.tlbr, 0, sc, img_id='generic')
        debugger.add_arrow(item.ct, item.tracking_offset, img_id='generic')
    
    if self.opt.debug >= 1:
      debugger.save_all_imgs(self.opt.debug_dir, prefix='{}'.format(self.cnt))

  def reset_tracking(self):
    self.tracker.reset()
    self.pre_images = None
    self.pre_image_ori = None
