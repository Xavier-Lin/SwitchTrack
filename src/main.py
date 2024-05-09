from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import math
from loguru import logger as lg

import torch
import torch.utils.data
import torch.distributed as dist

from lib.opts import opts
from lib.logger import Logger
from lib.trainer import Trainer
from lib.dataset.dataset_factory import get_dataset
from lib.model.model import create_model, save_model

from contextlib import contextmanager

def get_optimizer(opt, model):
  if opt.optim == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)# Adam do not use LR-schedule
  elif opt.optim == 'sgd':
    lg.info('Using SGD')
    optimizer = torch.optim.SGD(
      model.parameters(), opt.lr, momentum=0.9, weight_decay=0.0001, nesterov=True)
  else:
    assert 0, opt.optim
  return optimizer

@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()

def main(opt):
  # use seed
  torch.manual_seed(opt.seed)
  
  # cudnn acceleration
  torch.backends.cudnn.benchmark = True
  start_epoch = 0
  
  # set device and batch per GPU
  n_gpus = torch.cuda.device_count()
  opt.batch_size_per_gpu = opt.batch_size // n_gpus
  torch.cuda.set_device(opt.local_rank) 
  opt.device = torch.device('cuda', opt.local_rank)

  # create model and optimizer
  lg.info('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)
  optimizer = get_optimizer(opt, model)#  optimizer without loss parameter
  
  # build dataset and train dataloader 
  lg.info('Setting up train data...')
  with torch_distributed_zero_first(opt.global_rank):
    Dataset = get_dataset(opt.dataset)
  train_dataset = Dataset(opt, 'train')
  train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
  train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batch_size_per_gpu, sampler=train_sampler, pin_memory=True, num_workers=opt.num_workers)

  # build trainer
  lg.info('Build trainer... ')
  trainer = Trainer(opt, model, optimizer)
    
  # resume epoch
  if opt.load_model != '' and opt.resume:
    start_epoch = trainer.resume_epoch
  
  # lr setting
  opt.lr_step = [int(eps) for eps in opt.lr_step.split(',')]
  
  #log opt
  logger = Logger(opt)
  lg.info('Starting training...')
  
  for epoch in range(start_epoch+1, opt.num_epochs+1):
    lg.info('Start epoch : {}'.format(epoch))
    log_dict_train, _ = trainer.train(epoch, train_loader)
    logger.write('epoch: {} |'.format(epoch))
    
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
      
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      if opt.global_rank in [-1, 0]:# save model in process 0 (half model)
        save_model(
          os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
          epoch, 
          trainer.model_with_loss, 
          trainer.optimizer
        )
        
    logger.write('\n')

    if epoch in opt.lr_step:
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in trainer.optimizer.param_groups:
          param_group['lr'] = lr
    
  logger.close()

if __name__ == '__main__':
  opt = opts().parse()
  opt = opts().update_res_info_and_set_heads(opt)
  opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1 
  opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1 
  dist.init_process_group(backend='nccl')
  main(opt)

