from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from utils.basetrack import BaseTrack, TrackState
from collections import deque
from utils.match import  embedding_distance, fuse_score_iou, linear_assignment, iou_distance, ct_distance_iou_match
class STrack(BaseTrack):
  def __init__(self, tlbr, cls_id, ct, tracking_offset, score, feat):
    # wait activate
    self.tlbr = tlbr
    self.cls_id = cls_id
    self.ct = ct
    self.tracking_offset = tracking_offset
    self.score = score

    self.is_activated = False
    self.tracklet_len = 0
    
    self.smooth_feat = None
    self.curr_feat = None
    if feat is not None:
        self.update_features(feat, score)
    self.features = deque([], maxlen=30)
    self.gamma = 0.4
    
  def update_features(self, feat, score = None):
    feat /= np.linalg.norm(feat)
    self.curr_feat = feat
    if self.smooth_feat is None:
      self.smooth_feat = feat
    else:
      self.smooth_feat = self.gamma * self.smooth_feat * (1 - score) + (1 - self.gamma) * feat * score
    self.features.append(feat)
    self.smooth_feat /= np.linalg.norm(self.smooth_feat)

  
  def activate(self, frame_id):
    """Start a new tracklet"""
    self.track_id = self.next_id()
    self.tracklet_len = 0
    self.state = TrackState.Tracked
    if frame_id == 1:
      self.is_activated = True
   
    self.frame_id = frame_id
    self.start_frame = frame_id

  def re_activate(self, new_track, frame_id, new_id=False):
    self.tlbr = new_track.tlbr
    self.ct = new_track.ct
    self.tracking_offset = new_track.tracking_offset
    self.cls_id = new_track.cls_id
    self.score = new_track.score
    if new_track.curr_feat is not None:
      self.update_features(new_track.curr_feat, new_track.score)

    self.tracklet_len = 0
    self.state = TrackState.Tracked
    self.is_activated = True
    self.frame_id = frame_id
    if new_id:
      self.track_id = self.next_id()

  def update(self, new_track, frame_id):
    """
    Update a matched track :type new_track: 
    """
    self.frame_id = frame_id
    self.tracklet_len += 1

    self.tlbr = new_track.tlbr
    self.ct = new_track.ct
    self.tracking_offset = new_track.tracking_offset
    self.cls_id = new_track.cls_id
    if new_track.curr_feat is not None:
      self.update_features(new_track.curr_feat, new_track.score)
    
    self.state = TrackState.Tracked
    self.is_activated = True

    self.score = new_track.score

  def __repr__(self):
    return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class Tracker(object):
  def __init__(self, opt):
    self.opt = opt
    self.max_time_lost = 30 # 30 ｜ 60
    self.reset()
      
  def reset(self):# reset tracker for every seq
    self.frame_id = 0
    self.tracks = [] # total tracklets pre_frame
    self.tracked_stracks = []  # type: list[STrack] 
    self.lost_stracks = []  # type: list[STrack]
    self.removed_stracks = []  # type: list[STrack]
    self.num_emb = 0

  def step(self, results):
    self.frame_id += 1
    activated_starcks = [] 
    refind_stracks = [] 
    lost_stracks = [] 
    removed_stracks = [] 

    if len(results):
      dets = [STrack(x['bbox'], x['class'], x['ct'], x['tracking'], x['score'], x['reid']) 
                        for x in results]
    else:
      dets = []
    
    unconfirmed = []
    tracked_stracks = []  # type: list[STrack]
    for track in self.tracked_stracks:
      if not track.is_activated:
        unconfirmed.append(track)#unact tracked
      else:
        tracked_stracks.append(track)#act tracked

    """ Step 1: Associating online tracks"""
    matches, u_track, u_det = ct_distance_iou_match(tracked_stracks, dets, 0.7, is_hungarian=self.opt.hungarian)# 0.5 ｜ 0.7
    for itracked, idet in matches:
      track = tracked_stracks[itracked]
      det =  dets[idet]
      if track.state == TrackState.Tracked:
        track.update(det, self.frame_id)# act_tracked - act_tracked  add to  activated_starcks.
        activated_starcks.append(track)

    for it in u_track:#  add to lost_stracks
      track = tracked_stracks[it]
      track.mark_lost()
      lost_stracks.append(track)

    """ Step 2: Associating lost tracks """
    dets = [dets[i] for i in u_det]
    dist = embedding_distance(self.lost_stracks, dets)
    matches, u_track, u_det  = linear_assignment(dist, thresh=0.2) # 0.17 ｜ 0.2
    self.num_emb += len(matches)
    for itracked, idet in matches:
      track =self.lost_stracks[itracked]
      det =  dets[idet]
      if track.state == TrackState.Lost:
        track.re_activate(det, self.frame_id, new_id=False)# act_tracked - act_tracked  add to  activated_starcks.
        refind_stracks.append(track)
    
      
    """ Step 3: Third matching, deal with new tracklets from previous frame """
    dets = [dets[i] for i in u_det]
    dists = iou_distance(unconfirmed, dets)
    dists = fuse_score_iou(dists, dets)
    matches, u_unconfirmed, u_det = linear_assignment(dists, thresh=0.6)
    for itracked, idet in matches:
      unconfirmed[itracked].update(dets[idet], self.frame_id)
      activated_starcks.append(unconfirmed[itracked])
    for it in u_unconfirmed:
      track = unconfirmed[it]
      track.mark_removed()
      removed_stracks.append(track)

    
    """ Step 4: Init new stracks of current frame """
    for inew in u_det:
      track = dets[inew]
      if track.score < self.opt.new_thresh: # 0.65 ｜ 0.8
        continue
      track.activate(self.frame_id)
      activated_starcks.append(track)

    """ Step 5: Remove Track which exceed max_age """
    for track in self.lost_stracks:
      # self.lost_stracks = refind + act_lost
      if self.frame_id - track.end_frame > self.max_time_lost:
        track.mark_removed()
        removed_stracks.append(track)
    # self.lost_stracks  = refind + act_lost + removed 

    """ Summary... """
    self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
    self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
    self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
    self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
    self.lost_stracks.extend(lost_stracks)
    self.lost_stracks = sub_stracks(self.lost_stracks, removed_stracks)
    self.removed_stracks.extend(removed_stracks)
    self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
    output_stracks = [track for track in self.tracked_stracks if track.is_activated]
    self.tracks = output_stracks
    
    return output_stracks
 

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res

def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())

def remove_duplicate_stracks(stracksa, stracksb):
    pdist = iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb



