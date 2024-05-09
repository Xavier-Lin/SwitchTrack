from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy
from scipy.spatial.distance import cdist
import lap
import copy
from cython_bbox import bbox_overlaps as bbox_ious

def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b
            
def fuse_score_iou(cost_matrix, detections):
  if cost_matrix.size == 0:
    return cost_matrix
  iou_sim = 1 - cost_matrix
  det_scores = np.array([det.score for det in detections])
  det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
  fuse_sim = iou_sim * det_scores
  fuse_cost = 1 - fuse_sim
  return fuse_cost

def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric)) # Nomalized features

    return cost_matrix

def ct_distance_match(tks, dets, is_hungarian=False):
  pre_cts = np.array(
    [det.ct + det.tracking_offset for det in dets], np.float32)

  track_size = np.array([((track.tlbr[2] - track.tlbr[0]) * \
    (track.tlbr[3] - track.tlbr[1])) \
    for track in tks], np.float32)

  track_cat = np.array([track.cls_id for track in tks], np.int32)  

  item_size = np.array([((item.tlbr[2] - item.tlbr[0]) * \
    (item.tlbr[3] - item.tlbr[1])) \
    for item in dets], np.float32)  
  

  item_cat = np.array([item.cls_id for item in dets], np.int32) 


  tracks = np.array(
    [pre_det.ct for pre_det in tks], np.float32) 
   

  dist = (((tracks.reshape(1, -1, 2) - \
                pre_cts.reshape(-1, 1, 2)) ** 2).sum(axis=2))
  
  invalid = ((dist > track_size.reshape(1, -1)) + \
    (dist > item_size.reshape(-1, 1)) + \
    (item_cat.reshape(-1, 1) != track_cat.reshape(1, -1))) > 0 
  dist = dist + invalid * 1e18 

  dist[dist > 1e18] = 1e18

  if is_hungarian:
    matches, u_track, u_det = linear_assignment(dist.T, 1e16)
  else:
    matches = greedy_assignment(copy.deepcopy(dist))
    u_det = [d for d in range(pre_cts.shape[0]) \
        if not (d in matches[:, 1])]
    u_track = [d for d in range(tracks.shape[0]) \
      if not (d in matches[:, 0])]
  
  return matches,  u_track,  u_det



def ct_distance_iou_match(tks, dets, thresh, is_hungarian=False):
  pre_cts = np.array(
    [det.ct + det.tracking_offset for det in dets], np.float32).reshape(-1, 2)
  
  pre_whs = np.array(
    [[det.tlbr[2] - det.tlbr[0], det.tlbr[3]-det.tlbr[1]] for det in dets], np.float32).reshape(-1, 2)
  

  track_whs = np.array([
    [track.tlbr[2] - track.tlbr[0], track.tlbr[3] - track.tlbr[1]] for track in tks
    ], np.float32).reshape(-1, 2)


  track_cts = np.array(
    [pre_det.ct for pre_det in tks], np.float32).reshape(-1, 2) 
   
    
  pre_xywh = np.concatenate([pre_cts, pre_whs], axis=1) # cx cy w h
  trk_xywh = np.concatenate([track_cts, track_whs], axis=1) # cx cy w h
  
  pre_xyxy = cxcywh2xyxy(pre_xywh)
  trk_xyxy = cxcywh2xyxy(trk_xywh)
  if trk_xyxy.shape[0] ==0 or pre_xyxy.shape[0] ==0:
    dist = np.array([], np.int32).reshape(trk_xyxy.shape[0], pre_xyxy.shape[0])
  else:
    _ious = ious(trk_xyxy, pre_xyxy)
    dist = 1 - _ious
    # dist = fuse_score_iou(dist, dets)

  if is_hungarian:
    matches, u_track, u_det = linear_assignment(dist, thresh)
  else:
    matches = greedy_assignment_iou(copy.deepcopy(dist.T), thresh)
    u_det = [d for d in range(pre_xyxy.shape[0]) \
        if not (d in matches[:, 1])]
    u_track = [d for d in range(trk_xyxy.shape[0]) \
      if not (d in matches[:, 0])]
  
  return matches,  u_track,  u_det


def cxcywh2xyxy(bbox):
  x1 = bbox[:, 0:1] - bbox[:, 2:3]/2 # x1
  y1 = bbox[:, 1:2] - bbox[:, 3:]/2 # y1
  x2 =  x1 + bbox[:, 2:3] # x2
  y2 =  y1 + bbox[:, 3:]# y2
  xyxy = np.concatenate([x1,y1,x2,y2], axis=1)
  return xyxy



def greedy_assignment(dist):
  matched_indices = []
  if dist.shape[1] == 0:
    return np.array(matched_indices, np.int32).reshape(-1, 2)
  for i in range(dist.shape[0]):
    j = dist[i].argmin()
    if dist[i][j] < 1e16:
      dist[:, j] = 1e18
      matched_indices.append([j, i])
  return np.array(matched_indices, np.int32).reshape(-1, 2)

def greedy_assignment_iou(dist, thresh):
  matched_indices = []
  if dist.shape[1] == 0:
    return np.array(matched_indices, np.int32).reshape(-1, 2)
  for i in range(dist.shape[0]):
    j = dist[i].argmin()
    if dist[i][j] < thresh:
      dist[:, j] = 1.
      matched_indices.append([j, i])
  return np.array(matched_indices, np.int32).reshape(-1, 2)

def ious(atlbrs, btlbrs):
  """
  Compute cost based on IoU
  :type atlbrs: list[tlbr] | np.ndarray
  :type atlbrs: list[tlbr] | np.ndarray
  :rtype ious np.ndarray
  """
  ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
  if ious.size == 0:
      return ious

  ious = bbox_ious(
      np.ascontiguousarray(atlbrs, dtype=np.float),
      np.ascontiguousarray(btlbrs, dtype=np.float)
  )

  return ious


def iou_distance(atracks, btracks):
  """
  Compute cost based on IoU  
  :type atracks: list[STrack]
  :type btracks: list[STrack]
  :rtype cost_matrix np.ndarray
  """
  atlbrs = [track.tlbr for track in atracks]
  btlbrs = [track.tlbr for track in btracks]
  _ious = ious(atlbrs, btlbrs)
  cost_matrix = 1 - _ious

  return cost_matrix 

