from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from bar import Bar


from utils.utils import AverageMeter
from model.losses import FastFocalLoss, RegWeightedL1Loss, FocalLoss, IOUloss, ReidLoss
from model.model import load_model
from model.decode import generic_decode
from model.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import generic_post_process
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from loguru import logger as lg

class GenericLoss(torch.nn.Module):
  def __init__(self, opt):
    super(GenericLoss, self).__init__()
    # focal_loss used for heatmap
    if opt.focalLoss == 'FastFocalLoss':
      self.crit = FastFocalLoss(opt=opt)
    elif opt.focalLoss == 'FocalLoss':
      self.crit = FocalLoss()
      
    # l1 loss used for reg
    self.crit_reg = RegWeightedL1Loss()
    
    # iou loss for bbox
    if opt.iouloss:
      if opt.local_rank in [-1,0]:
        lg.info('using iouloss now !')
      self.iouloss = IOUloss(opt)
      
    # reid loss
    if opt.reid:
      self.idloss = ReidLoss(opt=opt)

    self.opt = opt

  def _sigmoid_output(self, output):
    if 'hm' in output:
      output['hm'] = _sigmoid(output['hm'])
    return output

  def forward(self, outputs, batch):
    opt = self.opt
    losses = {head: 0 for head in opt.heads}
    bz = batch['image'].shape[0]
  
    if opt.iouloss:
      losses['iou'] = 0 
      iou_weight = 2 

    for s in range(opt.num_stacks):
      output = outputs[s]# output = {'hm':, 'reg':, 'tracking':, 'ltrb_amodal':, 'reid':}
      output = self._sigmoid_output(output)

      if 'hm' in output:# fastfocalloss
        if opt.focalLoss == 'FastFocalLoss':
          losses['hm'] += self.crit(
            output['hm'], batch['hm'], batch['ind'], 
            batch['mask'], batch['cat']) / opt.num_stacks
        elif opt.focalLoss == 'FocalLoss':
          losses['hm'] += self.crit(
            output['hm'], batch['hm']) / opt.num_stacks

      if opt.iouloss:
        losses['iou'] += self.iouloss(
          output['ltrb_amodal'], batch['ltrb_amodal_mask'], batch['ind'], batch['bbox_amodal'], batch['hm']) / opt.num_stacks
      
      if opt.reid and 'reid' in output:
        # output['reid'] batch['ind']  batch['reid_mask']  batch['reid']
        losses['reid'] += self.idloss(output['reid'], batch['ind'],  batch['reid_mask'],  batch['reid']) / opt.num_stacks
        # nn.CrossEntropyLoss(ignore_index=-1)

      regression_heads = [
        'reg', 'tracking', 'ltrb_amodal']

      for head in regression_heads:
        if head in output:
          losses[head] += self.crit_reg(
            output[head], batch[head + '_mask'],
            batch['ind'], batch[head]) / opt.num_stacks

    losses['total_loss'] = 0
    for head in opt.heads:
      losses['total_loss'] += opt.weights[head] * losses[head]

    if opt.iouloss is True:
      losses['total_loss'] = iou_weight * losses['iou'] + losses['total_loss']

    return losses['total_loss']*bz, losses


class ModleWithLoss(torch.nn.Module):
  def __init__(self, model, loss):
    super(ModleWithLoss, self).__init__()
    self.model = model
    self.loss = loss
  
  def forward(self, batch):
    pre_img = batch['pre_img'] if 'pre_img' in batch else None
    pre_hm = batch['pre_hm'] if 'pre_hm' in batch else None
    outputs = self.model(batch['image'], pre_img, pre_hm)
    loss, loss_stats = self.loss(outputs, batch)
    return outputs[-1], loss, loss_stats
  

class Trainer(object):
  def __init__(self, opt, model):
    self.opt = opt
    
    # Get loss
    self.loss_stats, loss_func = self._get_losses(opt)
    
    if not opt.iouloss:
      if opt.local_rank in [-1,0]:
        lg.info(
          'loss_item :{}'.format(self.loss_stats) )
    else:
      if opt.local_rank in [-1,0]:
        lg.info(
          'loss_item :{}'.format(self.loss_stats+['iou']) )
    # Build model
    model_w_l = ModleWithLoss(model, loss_func)
    # model to device
    self.model_with_loss = model_w_l.to(f'cuda:{opt.local_rank}')
    
    # import pdb;pdb.set_trace()
    # for i,(k,v) in enumerate(self.model_with_loss.named_parameters()):
    #   print(f'{i}-{k}')
    # import pdb;pdb.set_trace()
    
    # Build optimizer
    self.optimizer = self.get_optimizer()#  optimizer with loss parameter
    
    # Resume
    if opt.load_model != '' and opt.resume:
      self.model_with_loss, self.optimizer, self.resume_epoch = load_model(
        self.model_with_loss, opt.load_model, opt, self.optimizer
      )
      if opt.local_rank in [-1,0]:
        lg.info('resuming epoch step : {}.'.format(self.resume_epoch))
    
    # DDP
    self.model_with_loss = DDP(
      self.model_with_loss, device_ids=[self.opt.local_rank])
          
  def get_optimizer(self):
    if self.opt.optim == 'adam':
      optimizer = torch.optim.Adam(self.model_with_loss.parameters(), self.opt.lr)# Adam do not use LR-schedule
    elif self.opt.optim == 'sgd':
      lg.info('Using SGD')
      optimizer = torch.optim.SGD(
        self.model_with_loss.parameters(), self.opt.lr, momentum=0.9, weight_decay=0.0001, nesterov=True)
    else:
      assert 0, self.opt.optim
    return optimizer

  def run_epoch(self, epoch, data_loader):# 输入‘train’  eps  train_loader
    opt = self.opt
    self.model_with_loss.train()
    
    results = {}
    if self.opt.local_rank in [-1,0]:
      data_time, batch_time = AverageMeter(), AverageMeter()
      avg_loss_stats = {l: AverageMeter() for l in self.loss_stats  if l == 'total_loss' or opt.weights[l] > 0 }
      if opt.iouloss:
        avg_loss_stats['iou'] = AverageMeter()
    num_iters = len(data_loader)
    if self.opt.local_rank in [-1,0]:
      bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
      end = time.time()
    # dataset.random_short_size()
    for iter_id, batch in enumerate(data_loader):
      if iter_id >= num_iters:
        break
      # batch ={'image': curr_img after aug,
      #       'pre_img': pre_img after the same aug,
      #       'pre_hm': Previous frame GT heat map with network input size
      #       'hm': shape(1, h/4, w/4),-- curr_img heat map
      #       'ind': shape(K,), 'cat':shape(K,), 'mask':shape(K,), 
      #       'reg':shape(k, 2), 'reg_mask':shape(k, 2), 
      #       'tracking': shape(k, 2), 'tracking_mask':shape(k, 2),
      #       'ltrb_amodal': shape(k, 4), 'ltrb_amodal_mask':shape(k, 4)
      #       'reid': shape(k,),  'reid_mask': shape(k,)}
      if self.opt.local_rank in [-1,0]:
        data_time.update(time.time() - end)

      for k in batch:
        if k != 'meta':
          batch[k] = batch[k].cuda(device=opt.local_rank, non_blocking=True)
         
      # Forward
      output, loss, loss_stats = self.model_with_loss(batch)
      
      torch.distributed.barrier()
      # Backward
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
      
      if self.opt.local_rank in [-1,0]:
        batch_time.update(time.time() - end)
        end = time.time()

        Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
          epoch, iter_id, num_iters, phase='train',
          total=bar.elapsed_td, eta=bar.eta_td)
        
        for l in avg_loss_stats:
          avg_loss_stats[l].update(
            loss_stats[l].mean().item() if not isinstance(loss_stats[l], float) else loss_stats[l], batch['image'].size(0))
          Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
        
        Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
          '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
        
        bar.next()
      
      if opt.debug > 0:
        self.debug(batch, output, iter_id, dataset=data_loader.dataset)
      del output, loss, loss_stats
      # dataset.random_short_size()
      
    # end iter
    if self.opt.local_rank in [-1,0]:
      bar.finish()
      ret = {k: v.avg for k, v in avg_loss_stats.items()}
      ret['time'] = bar.elapsed_td.total_seconds() / 60.
    else:
      ret ={}
    return ret, results
  
  def _get_losses(self, opt):
    loss_order = ['hm', 'reg', 'ltrb_amodal', 'tracking', 'reid']
    loss_states = ['total_loss'] + [k for k in loss_order if k in opt.heads]
    loss = GenericLoss(opt)
    return loss_states, loss
  
  def get_lr_scheduler(self):
    pass

  def debug(self, batch, output, iter_id, dataset):
    opt = self.opt
    if 'pre_hm' in batch:
      output.update({'pre_hm': batch['pre_hm']})
    dets = generic_decode(output, K=opt.K, opt=opt)
    for k in dets:
      dets[k] = dets[k].detach().cpu().numpy()
    dets_gt = batch['meta']['gt_det']
    for i in range(1):
      debugger = Debugger(opt=opt, dataset=dataset)
      img = batch['image'][i].detach().cpu().numpy().transpose(1, 2, 0)
      img = np.clip(((
        img * dataset.std + dataset.mean) * 255.), 0, 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm')
      debugger.add_blend_img(img, gt, 'gt_hm')

      if 'pre_img' in batch:
        pre_img = batch['pre_img'][i].detach().cpu().numpy().transpose(1, 2, 0)
        pre_img = np.clip(((
          pre_img * dataset.std + dataset.mean) * 255), 0, 255).astype(np.uint8)
        debugger.add_img(pre_img, 'pre_img_pred')
        debugger.add_img(pre_img, 'pre_img_gt')
        if 'pre_hm' in batch:
          pre_hm = debugger.gen_colormap(
            batch['pre_hm'][i].detach().cpu().numpy())
          debugger.add_blend_img(pre_img, pre_hm, 'pre_hm')

      debugger.add_img(img, img_id='out_pred')
      if 'ltrb_amodal' in opt.heads:
        debugger.add_img(img, img_id='out_pred_amodal')
        debugger.add_img(img, img_id='out_gt_amodal')

      # Predictions
      for k in range(len(dets['scores'][i])):
        if dets['scores'][i, k] > opt.vis_thresh:
          debugger.add_coco_bbox(
            dets['bboxes'][i, k] * opt.down_ratio, dets['clses'][i, k],
            dets['scores'][i, k], img_id='out_pred')

          if 'ltrb_amodal' in opt.heads:
            debugger.add_coco_bbox(
              dets['bboxes_amodal'][i, k] * opt.down_ratio, dets['clses'][i, k],
              dets['scores'][i, k], img_id='out_pred_amodal')

          if 'hps' in opt.heads and int(dets['clses'][i, k]) == 0:
            debugger.add_coco_hp(
              dets['hps'][i, k] * opt.down_ratio, img_id='out_pred')

          if 'tracking' in opt.heads:
            debugger.add_arrow(
              dets['cts'][i][k] * opt.down_ratio, 
              dets['tracking'][i][k] * opt.down_ratio, img_id='out_pred')
            debugger.add_arrow(
              dets['cts'][i][k] * opt.down_ratio, 
              dets['tracking'][i][k] * opt.down_ratio, img_id='pre_img_pred')

      # Ground truth
      debugger.add_img(img, img_id='out_gt')
      for k in range(len(dets_gt['scores'][i])):
        if dets_gt['scores'][i][k] > opt.vis_thresh:
          debugger.add_coco_bbox(
            dets_gt['bboxes'][i][k] * opt.down_ratio, dets_gt['clses'][i][k],
            dets_gt['scores'][i][k], img_id='out_gt')

          if 'ltrb_amodal' in opt.heads:
            debugger.add_coco_bbox(
              dets_gt['bboxes_amodal'][i, k] * opt.down_ratio, 
              dets_gt['clses'][i, k],
              dets_gt['scores'][i, k], img_id='out_gt_amodal')

          if 'hps' in opt.heads and \
            (int(dets['clses'][i, k]) == 0):
            debugger.add_coco_hp(
              dets_gt['hps'][i][k] * opt.down_ratio, img_id='out_gt')

          if 'tracking' in opt.heads:
            debugger.add_arrow(
              dets_gt['cts'][i][k] * opt.down_ratio, 
              dets_gt['tracking'][i][k] * opt.down_ratio, img_id='out_gt')
            debugger.add_arrow(
              dets_gt['cts'][i][k] * opt.down_ratio, 
              dets_gt['tracking'][i][k] * opt.down_ratio, img_id='pre_img_gt')

      if opt.debug == 4:
        debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))

  def train(self, epoch, data_loader):
    return self.run_epoch(epoch, data_loader)
