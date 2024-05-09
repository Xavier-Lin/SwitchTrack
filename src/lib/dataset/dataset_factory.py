from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

from .datasets.coco import COCO
from .datasets.coco_hp import COCOHP
from .datasets.mot import MOT
from .datasets.crowdhuman import CrowdHuman
from .datasets.custom_dataset import CustomDataset

dataset_factory = {
  'custom': CustomDataset,
  'coco': COCO,
  'coco_hp': COCOHP,
  'mot': MOT,
  'crowdhuman': CrowdHuman,
}


def get_dataset(dataset):
  return dataset_factory[dataset]
