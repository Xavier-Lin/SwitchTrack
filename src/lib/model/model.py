from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch
import os
from loguru import logger as lg
from .networks.dla import DLASeg

_network_factory = {
  'dla': DLASeg,
}

def create_model(arch, head, head_conv, opt=None):
  num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0 # dla_34 
  arch = arch[:arch.find('_')] if '_' in arch else arch # ‘dla’
  model_class = _network_factory[arch]# model_class == DLASeg
  model = model_class(num_layers, heads=head, head_convs=head_conv, opt=opt)
  return model

def load_model(model, model_path, opt, optimizer=None):
  '''
  Args:
    model: created training model.
    model_path: the ckpt file path.
    opt: the args of train settings.
    optimizer: inital optimizer.
  '''
  checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
  # vis
  # for kkk in checkpoint['state_dict']:
  #   print(kkk)
  #checkpoint = {'epoch':, 'state_dict':,  'optimizer':, }
  lg.info('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
  state_dict_ = checkpoint['state_dict']
  state_dict = {}
  
  # convert data_parallal to model
  for k in state_dict_:
    if k.startswith('module') and not k.startswith('module_list'):
      state_dict[k[7:]] = state_dict_[k]
    elif k.startswith('model') and optimizer is None:
      state_dict[k[6:]] = state_dict_[k]
    else:
      state_dict[k] = state_dict_[k]
  model_state_dict = model.state_dict()
 
  # check loaded parameters and created model parameters
  drop_k = None; drop_idx = []; drop_num = 0
  for k in state_dict:# ckpt
    # search the information of drop parameters load from ckpt 
    if drop_k == k.split('.')[:-1]:
      pass
    else:
      drop_k = k.split('.')[:-1]
      drop_num += 1
      
    if k in model_state_dict: # model with loss
      if (state_dict[k].shape != model_state_dict[k].shape) or \
        (opt.reset_hm and k.startswith('hm') and (state_dict[k].shape[0] in [80, 1])):
        if opt.reuse_hm:
          lg.info('Reusing parameter {}, required shape {}, '\
                'loaded shape {}.'.format(
            k, model_state_dict[k].shape, state_dict[k].shape))
          if state_dict[k].shape[0] < state_dict[k].shape[0]:
            model_state_dict[k][:state_dict[k].shape[0]] = state_dict[k]
          else:
            model_state_dict[k] = state_dict[k][:model_state_dict[k].shape[0]]
          state_dict[k] = model_state_dict[k]
        else:
          lg.info('Skip loading parameter {}, required shape {}, '\
                'loaded shape {}.'.format(
            k, model_state_dict[k].shape, state_dict[k].shape))
          state_dict[k] = model_state_dict[k]
    else:
      drop_idx.append(drop_num-1)
      lg.info('Drop parameter: {}, drop_idx: {} !'.format(k, drop_num-1))
  
  for k in model_state_dict:
    if not (k in state_dict):
      lg.info('No param {}.'.format(k))
      state_dict[k] = model_state_dict[k]
  model.load_state_dict(state_dict, strict=False)
  
  # resume optimizer parameters
  if optimizer is not None and opt.resume:
    if 'optimizer' in checkpoint:
      ckpt_epoch = checkpoint['epoch']
      ckpt_lr_group = []
      
      # Reload the lr of ckpt file
      for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = ((checkpoint['optimizer'])['param_groups'][i])['lr']
        ckpt_lr_group.append(
          ((checkpoint['optimizer'])['param_groups'][i])['lr'])
      lg.info('Reloading optimizer with ckpt lr: {}'.format(ckpt_lr_group))
        
      # Reload state values of optimizer of ckpt file, excluding state values of drop parameters
      sign = 0; new_k = 0
      if len(checkpoint['optimizer']['state']) != 0:
        for k in checkpoint['optimizer']['state']:
          if k in drop_idx:
            sign += 1
            continue
          elif sign >= 1:
            new_k = k - sign
            optimizer.state.setdefault(new_k, checkpoint['optimizer']['state'][k])
          else:
            optimizer.state.setdefault(k, checkpoint['optimizer']['state'][k])
        lg.info('Reloading the state values of optimizer successfully !')
    else:
      lg.info('No optimizer parameters in checkpoint.')

  # optimizer.state_dict() -- ['state', 'param_groups']
  
  if optimizer is not None:# training 
    return model, optimizer, ckpt_epoch
  else:
    return model

def save_model(path, epoch, model, optimizer=None):
  '''
  path, 
  epoch, 
  trainer.model_with_loss, 
  trainer.optimizer

  '''
  if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
    model_state_dict = model.module.state_dict()
  else:
    model_state_dict = model.state_dict()
    
  data = {'epoch': epoch, 'state_dict': model_state_dict}
  
  if not (optimizer is None):
    data['optimizer'] = optimizer.state_dict()
  torch.save(data, path)

class BatchNormXd(torch.nn.modules.batchnorm._BatchNorm):
  def _check_input_dim(self, input):
      # The only difference between BatchNorm1d, BatchNorm2d, BatchNorm3d, etc
      # is this method that is overwritten by the sub-class
      # This original goal of this method was for tensor sanity checks
      # If you're ok bypassing those sanity checks (eg. if you trust your inference
      # to provide the right dimensional inputs), then you can just use this method
      # for easy conversion from SyncBatchNorm
      # (unfortunately, SyncBatchNorm does not store the original class - if it did
      #  we could return the one that was originally created)
      return

def revert_sync_batchnorm(module):
  # this is very similar to the function that it is trying to revert:
  # https://github.com/pytorch/pytorch/blob/c8b3686a3e4ba63dc59e5dcfe5db3430df256833/torch/nn/modules/batchnorm.py#L679
  module_output = module
  if isinstance(module, torch.nn.modules.batchnorm.SyncBatchNorm):
    new_cls = BatchNormXd
    module_output = BatchNormXd(module.num_features,
                                            module.eps, module.momentum,
                                            module.affine,
                                            module.track_running_stats)
    if module.affine:
      with torch.no_grad():
        module_output.weight = module.weight
        module_output.bias = module.bias
    module_output.running_mean = module.running_mean
    module_output.running_var = module.running_var
    module_output.num_batches_tracked = module.num_batches_tracked
    if hasattr(module, "qconfig"):
      module_output.qconfig = module.qconfig
  for name, child in module.named_children():
    module_output.add_module(name, revert_sync_batchnorm(child))
  del module
  return module_output