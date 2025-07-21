from collections import OrderedDict
from pathlib import Path
import os
import pickle

import torch
import cv2
import torchvision
import torchreid
import numpy as np

from external.adaptors.fastreid_adaptor import FastReID


class EmbeddingComputer:
    def __init__(self, opt, dataset):
        self.crop_size = (128, 384)
        self.max_batch = opt.K
        # Only used for the general ReID model (not FastReID)
        self.normalize = False
        if dataset == "mot17":
            path = "external/weights/mot17_sbs_S50.pth"
        elif dataset == "mot20":
            path = "external/weights/mot20_sbs_S50.pth"
        elif dataset == "dance":
            path = "external/weights/dance_sbs_S50.pth"
        elif dataset == "sports":
            path = "external/weights/SportsMOT/sbs_S50/model_0058.pth"
        else:
            raise RuntimeError("Need the path for a new ReID model.")
        self.reid_model = FastReID(path)

    def compute_embedding(self, img, bbox):
        # Generate all of the patches
        crops = []
        # Basic embeddings
        h, w = img.shape[2:]
        results = torch.round(bbox).astype(torch.int32)
        # print(results)

        results[:, 0] = results[:, 0].clamp(0, w)
        results[:, 1] = results[:, 1].clamp(0, h)
        results[:, 2] = results[:, 2].clamp(0, w)
        results[:, 3] = results[:, 3].clamp(0, h)

        crops = []
        for p in results:
            crop = img[p[1] : p[3], p[0] : p[2]]
            # print(p)
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = cv2.resize(crop, self.crop_size, interpolation=cv2.INTER_LINEAR).astype(np.float32)
            if self.normalize:
                crop /= 255
                crop -= np.array((0.485, 0.456, 0.406))
                crop /= np.array((0.229, 0.224, 0.225))
            crop = torch.as_tensor(crop.transpose(2, 0, 1))
            crop = crop.unsqueeze(0)
            crops.append(crop)
        crops = torch.cat(crops, dim=0)

        # Create embeddings and l2 normalize them
        embs = []
        for idx in range(0, len(crops), self.max_batch):
            batch_crops = crops[idx : idx + self.max_batch]
            batch_crops = batch_crops.cuda()
            with torch.no_grad():
                batch_embs = self.model(batch_crops)
            embs.extend(batch_embs)
        embs = torch.stack(embs)
        embs = torch.nn.functional.normalize(embs, dim=-1)

        if not self.grid_off:
            embs = embs.reshape(bbox.shape[0], -1, embs.shape[-1])
        embs = embs.cpu().numpy()
        return embs