from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

def _sigmoid12(x):
  y = torch.clamp(x.sigmoid_(), 1e-12)
  return y

def _gather_feat(feat, ind):
  dim = feat.size(2)
  ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)# b,100,1
  feat = feat.gather(1, ind) 
  return feat 
# torch.gather(src, dim= , index= )
def _tranpose_and_gather_feat(feat, ind):
  feat = feat.permute(0, 2, 3, 1).contiguous()# b，h/4，w/4，2
  feat = feat.view(feat.size(0), -1, feat.size(3))# b， s， 2
  feat = _gather_feat(feat, ind)
  return feat

def flip_tensor(x):
  return torch.flip(x, [3])
  # tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  # return torch.from_numpy(tmp).to(x.device)

def flip_lr(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)

def flip_lr_off(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  tmp = tmp.reshape(tmp.shape[0], 17, 2, 
                    tmp.shape[2], tmp.shape[3])
  tmp[:, :, 0, :, :] *= -1
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)

def _nms(heat, kernel=3):#'hm':(b,1,h/4,w/4)  
  pad = (kernel - 1) // 2

  hmax = nn.functional.max_pool2d(
      heat, (kernel, kernel), stride=1, padding=pad)
  keep = (hmax == heat).float()
  return heat * keep

def _topk_channel(scores, K=100):
  batch, cat, height, width = scores.size()
  
  topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

  topk_inds = topk_inds % (height * width)
  topk_ys   = (topk_inds / width).int().float()
  topk_xs   = (topk_inds % width).int().float()

  return topk_scores, topk_inds, topk_ys, topk_xs

def _topk(scores, K=100):
  batch, cat, height, width = scores.size()# the size of heat-map
    
  topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)
  #  b c 100
  topk_inds = topk_inds % (height * width)# 对所有（100个） 
  topk_ys   = (topk_inds / width).int().float()
  topk_xs   = (topk_inds % width).int().float() 
    
  topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
  topk_clses = (topk_ind / K).int()
  topk_inds = _gather_feat(
      topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
  topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
  topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

  return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

def fuse_conv_and_bn(conv, bn):
  # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
  fusedconv = (
    nn.Conv2d(
      conv.in_channels,
      conv.out_channels,
      kernel_size=conv.kernel_size,
      stride=conv.stride,
      padding=conv.padding,
      groups=conv.groups,
      bias=True,
    )
    .requires_grad_(False)
    .to(conv.weight.device)
  )

  # prepare filters
  w_conv = conv.weight.clone().view(conv.out_channels, -1)
  w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
  fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

  # prepare spatial bias
  b_conv = (
      torch.zeros(conv.weight.size(0), device=conv.weight.device)
      if conv.bias is None
      else conv.bias
  )
  b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(
      torch.sqrt(bn.running_var + bn.eps)
  )
  fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

  return fusedconv


def fuse_model(model):
  from .networks.dla import BasicBlock, Root, DLA, Tree, Bottleneck
  from .networks.base_model import BFL

  for m in model.modules():
    if type(m) is BasicBlock:
      m.conv1 = fuse_conv_and_bn(m.conv1, m.bn1)  # update conv
      m.conv2 = fuse_conv_and_bn(m.conv2, m.bn2)
      delattr(m, "bn1")  # remove batchnorm
      delattr(m, "bn2")  # remove batchnorm
      m.forward = m.fuseforward  # update forward
    elif type(m) is Bottleneck:
      m.conv1 = fuse_conv_and_bn(m.conv1, m.bn1)  # update conv
      m.conv2 = fuse_conv_and_bn(m.conv2, m.bn2)
      m.conv3 = fuse_conv_and_bn(m.conv3, m.bn3)
      delattr(m, "bn1")  # remove batchnorm
      delattr(m, "bn2")  # remove batchnorm
      delattr(m, "bn3")  # remove batchnorm
      m.forward = m.fuseforward  # update forward
    elif type(m) is Root:
      m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
      delattr(m, "bn")  # remove batchnorm
      m.forward = m.fuseforward  # update forward
    elif type(m) is DLA:
      m.level0[0] = fuse_conv_and_bn(m.level0[0], m.level0[1])  # update conv
      delattr(m.level0, "1")  # remove batchnorm
      m.level1[0] = fuse_conv_and_bn(m.level1[0], m.level1[1])  # update conv
      delattr(m.level1, "1")  # remove batchnorm
      m.base_layer[0] = fuse_conv_and_bn(m.base_layer[0], m.base_layer[1])  # update conv
      delattr(m.base_layer, "1")  # remove batchnorm
      m.pre_img_layer[0] = fuse_conv_and_bn(m.pre_img_layer[0], m.pre_img_layer[1])  # update conv
      delattr(m.pre_img_layer, "1")  # remove batchnorm
      m.pre_hm_layer[0] = fuse_conv_and_bn(m.pre_hm_layer[0], m.pre_hm_layer[1])  # update conv
      delattr(m.pre_hm_layer, "1")  # remove batchnorm
      m.forward = m.fuseforward  # update forward
    elif type(m) is Tree:
      if hasattr(m, "project") and (m.project is not None):
        m.project[0] = fuse_conv_and_bn(m.project[0], m.project[1])  # update conv
        delattr(m.project, "1")  # remove batchnorm
        m.forward = m.fuseforward  # update forward
    elif type(m) is BFL:
      m.conv4 = fuse_conv_and_bn(m.conv4, m.bn4)  # update conv
      delattr(m, "bn4")  # remove batchnorm
      m.forward = m.fuseforward  # update forward
        
  return model


def replace_module(module, replaced_module_type, new_module_type, replace_func=None):
    """
    Replace given type in module to a new type. mostly used in deploy.

    Args:
        module (nn.Module): model to apply replace operation.
        replaced_module_type (Type): module type to be replaced.
        new_module_type (Type)
        replace_func (function): python function to describe replace logic. Defalut value None.

    Returns:
        model (nn.Module): module that already been replaced.
    """

    def default_replace_func(replaced_module_type, new_module_type):
        return new_module_type()

    if replace_func is None:
        replace_func = default_replace_func

    model = module
    if isinstance(module, replaced_module_type):
        model = replace_func(replaced_module_type, new_module_type)
    else:  # recurrsively replace
        for name, child in module.named_children():
            new_child = replace_module(child, replaced_module_type, new_module_type)
            if new_child is not child:  # child is already replaced
                model.add_module(name, new_child)

    return model