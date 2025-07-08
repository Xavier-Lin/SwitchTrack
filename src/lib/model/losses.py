# ------------------------------------------------------------------------------
# Portions of this code are from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from re import X
from turtle import forward

import torch
import torch.nn as nn
from .utils import _tranpose_and_gather_feat, _nms, _topk
import torch.nn.functional as F
from utils.image import draw_umich_gaussian
import math
import numpy as np
def _slow_neg_loss(pred, gt):
  '''focal loss from CornerNet'''
  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt[neg_inds], 4)

  loss = 0
  pos_pred = pred[pos_inds]
  neg_pred = pred[neg_inds]

  pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
  neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if pos_pred.nelement() == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss

def _neg_loss(pred, gt):
  ''' Reimplemented focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt, 4)

  loss = 0
  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()
  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss


def _only_neg_loss(pred, gt):
  gt = torch.pow(1 - gt, 4)
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * gt
  return neg_loss.sum()

class FastFocalLoss(nn.Module):
  '''
  Reimplemented focal loss, exactly the same as the CornerNet version.
  Faster and costs much less memory.
  '''
  def __init__(self, opt=None):
    super(FastFocalLoss, self).__init__()

  def forward(self, out, target, ind, mask, cat):
    '''
    Arguments:
      out, target: B x C x H x W 
      ind, mask: B x M
      cat (category id for peaks): B x M
    '''
    neg_loss = _only_neg_loss(out, target)
    pos_pred_pix = _tranpose_and_gather_feat(out, ind) # B x M x C 
    pos_pred = pos_pred_pix.gather(2, cat.unsqueeze(2)) # B x M
    num_pos = mask.sum()
    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2) * \
               mask.unsqueeze(2)
    pos_loss = pos_loss.sum()
    if num_pos == 0:
      return - neg_loss
    return - (pos_loss + neg_loss) / num_pos

def _reg_loss(regr, gt_regr, mask):
  ''' L1 regression loss
    Arguments:
      regr (batch x max_objects x dim)
      gt_regr (batch x max_objects x dim)
      mask (batch x max_objects)
  '''
  num = mask.float().sum()
  mask = mask.unsqueeze(2).expand_as(gt_regr).float()

  regr = regr * mask
  gt_regr = gt_regr * mask
    
  regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, reduction='sum')
  regr_loss = regr_loss / (num + 1e-4)
  return regr_loss


class RegWeightedL1Loss(nn.Module):
  def __init__(self):
    super(RegWeightedL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)#b num_objs 2
    mask = mask.float()
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss

#------ loss used in fairmot ------ 
class FocalLoss(nn.Module):
  '''nn.Module warpper for focal loss'''
  def __init__(self):
    super(FocalLoss, self).__init__()
    
  def forward(self, out, target):
    return _neg_loss(out, target)


class IOUloss(nn.Module):
  def __init__(self, opt):
    super(IOUloss, self).__init__()
    if opt.use_siou:
      self.siou = SIoU()
    self.opt = opt

  def forward(self, output, mask, ind, target, hm):
    # output['ltrb_amodal'],  batch['ltrb_amodal_mask'],  batch['ind'],  batch['bbox_amodal'],  batch['hm']
    pred = _tranpose_and_gather_feat(output, ind)
    b, K = pred.shape[:-1]
    h, w = hm.shape[2:]

    mask = mask.float()# b K 4
    pred_ = pred * mask
    target_ = target * mask # b K 4
    
    ys0   = (ind / w).int().float()# --- y
    xs0   = (ind % w).int().float() # ---- x
 
    pred_ = pred_.view(b, K, 4)
    bboxes_amodal = torch.cat([ xs0.view(b, K, 1) + pred_[..., 0:1], 
                                ys0.view(b, K, 1) + pred_[..., 1:2],
                                xs0.view(b, K, 1) + pred_[..., 2:3], 
                                ys0.view(b, K, 1) + pred_[..., 3:4]], dim=2) # b K 4 xyxy
    tar = target_[mask.sum(-1) > 0]
    pre = bboxes_amodal[mask.sum(-1) > 0]
    num=torch.sum(mask.sum(-1) > 0)
    if self.opt.use_siou:
      iou_loss = self.siou(pre, tar)
    else:
      iou_loss = self.iou(pre, tar)
    return iou_loss.sum() / (num + 1e-8)

  def iou(self, pred, target, reduction="none", loss_type="iou"):
    assert pred.shape[0] == target.shape[0]

    pred = pred.view(-1, 4)
    target = target.view(-1, 4)
    tl = torch.max( pred[:, :2], target[:, :2] )
    br = torch.min( pred[:, 2:], target[:, 2:] )

    area_p = (pred[:, 2] - pred[:, 0])*(pred[:, 3] - pred[:, 1])
    area_g = (target[:, 2] - target[:, 0])*(target[:, 3] - target[:, 1])

    en = (tl < br).type(tl.type()).prod(dim=1)
    area_i = torch.prod(br - tl, 1) * en
    iou = (area_i) / (area_p + area_g - area_i + 1e-16)

    if loss_type == "iou":
      loss = 1 - iou ** 2
    elif loss_type == "giou":
      c_tl = torch.min(
          (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
      )
      c_br = torch.max(
          (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
      )
      area_c = torch.prod(c_br - c_tl, 1)
      giou = iou - (area_c - area_i) / area_c.clamp(1e-16)
      loss = 1 - giou.clamp(min=-1.0, max=1.0)

    if reduction == "mean":
      loss = loss.mean()
    elif reduction == "sum":
      loss = loss.sum()

    return loss

class ReidLoss(nn.Module):
  def __init__(self, opt):
    super().__init__()
    self.classifier = nn.Linear(opt.heads['reid'], opt.nid+1)
    self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')
    self.emb_scale = math.sqrt(2) * math.log(opt.nid)
  
  def forward(self, pred, ind, reid_mask, target):
    # output['reid'] batch['ind']  batch['reid_mask']  batch['reid']
    id_head = _tranpose_and_gather_feat(pred, ind)# b K 128
    id_head = id_head[reid_mask > 0].contiguous()#  num_gt_all 128
    id_head = self.emb_scale * F.normalize(id_head)
    id_target = target[reid_mask > 0]# num_gt_all
    id_output = self.classifier(id_head).contiguous()
    if len(id_target) > 0 and len(id_output) > 0: 
      reidloss = self.IDLoss(id_output, id_target) / len(id_target)
    else:
      reidloss = self.IDLoss(id_output, id_target) * len(id_target)
    return reidloss

class SIoU(nn.Module):
  def __init__(self, ):
    super().__init__()
  
  def forward(self, pred, target):
    # """pred: N X 4   target: N X 4  -- x y x y"""
    assert pred.shape[0] == target.shape[0]
    iou, en=self.iou_(pred, target)
    bbox_loss = 1 - iou + ( self.dist_loss(pred, target)*en + self.shape_loss(pred, target)*en ) / 2 
    return bbox_loss

  def get_angle(self, pred, target):
    pred_ct = (pred[:,0] + pred[:, 2]) / 2 , (pred[:,1] + pred[:, 3]) / 2
    target_ct = (target[:,0] + target[:, 2]) / 2 , (target[:,1] + target[:, 3]) / 2

    ch = torch.max(target_ct[1], pred_ct[1]) - torch.min(target_ct[1], pred_ct[1])
    cw = torch.max(target_ct[0], pred_ct[0]) - torch.min(target_ct[0], pred_ct[0])
    sigma=torch.pow(cw ** 2 + ch ** 2, 0.5)

    x1 = torch.abs(cw) / (sigma + 1e-16)
    x2 = torch.abs(ch) / (sigma + 1e-16)
    threshold = pow(2, 0.5) / 2
    sin_alpha = torch.where(x1 > threshold, x2, x1)
    return pred_ct, target_ct, sin_alpha

  def angle_loss(self, pred, target):
    pred_ct, target_ct, x = self.get_angle(pred, target)
    # L = 1 - 2 * torch.pow(torch.sin(torch.arcsin(x) - np.pi/4), 2)
    L = torch.cos(torch.arcsin(x) * 2 - np.pi / 2)
    return pred_ct, target_ct, L

  def dist_loss(self, pred, target):
    pred = pred.view(-1, 4)
    target = target.view(-1, 4)
    pred_ct, target_ct, L = self.angle_loss(pred, target)

    cw = torch.max(pred[:, 2], target[:, 2]) - torch.min(pred[:, 0], target[:, 0])
    ch = torch.max(pred[:, 3], target[:, 3]) - torch.min(pred[:, 1], target[:, 1])

    Px = ((target_ct[0] - pred_ct[0]) / (cw + 1e-16)) **2
    Py = ((target_ct[1] - pred_ct[1]) / (ch + 1e-16)) **2
    gamma = 2 - L 
    detla = (1 - torch.exp(-gamma * Px)) + (1 - torch.exp(-gamma * Py))
    return detla
  

  def shape_loss(self, pred, target):
    theta = 4
    pred = pred.view(-1, 4)
    target = target.view(-1, 4)

    pred_w=pred[:, 2] - pred[:, 0]
    pred_h=pred[:, 3] - pred[:, 1]
    target_w=target[:, 2] - target[:, 0]
    target_h=target[:, 3] - target[:, 1]

    omiga_w = (torch.abs(pred_w - target_w)) /  \
              (torch.max(pred_w, target_w) + 1e-16)
    omiga_h = (torch.abs(pred_h - target_h)) /  \
              (torch.max(pred_h, target_h) + 1e-16)

    omiga = torch.pow(1 - torch.exp(- omiga_w), theta) + torch.pow(1 - torch.exp(- omiga_h), theta)
    return omiga 

  def iou_(self, pred, target):
    pred = pred.view(-1, 4)
    target = target.view(-1, 4)

    tl = torch.max( pred[:, :2], target[:, :2] )
    br = torch.min( pred[:, 2:], target[:, 2:] )

    area_p = (pred[:, 2] - pred[:, 0])*(pred[:, 3] - pred[:, 1])
    area_g = (target[:, 2] - target[:, 0])*(target[:, 3] - target[:, 1])

    en = (tl < br).type(tl.type()).prod(dim=1)
    area_i = torch.prod(br - tl, 1) * en
    iou = torch.abs( (area_i) / (area_p + area_g - area_i + 1e-16) )
    return iou, en