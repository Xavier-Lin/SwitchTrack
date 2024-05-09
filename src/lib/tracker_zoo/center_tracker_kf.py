import numpy as np
# from scipy.optimize import linear_sum_assignment as linear_assignment
import lap
from tracking_utils.kalman import KalmanFilter
import copy
from utils.match import iou_distance, linear_assignment
from utils.basetrack import TrackState, BaseTrack
class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score):

        # wait activate
        # import pdb;pdb.set_trace()
        self._tlwh = tlwh
        
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
      
        self.score = score
        self.tracklet_len = 0


    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov
                

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        #self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        
    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score


    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)

class KFCTracker(object):
  def __init__(self, opt):
    self.opt = opt
    self.kalman_filter = KalmanFilter()
    self.reset()

  def reset(self):
    self.frame_id = 0
    self.tracks = []

  def step(self, results):
    
    N = len(results)
    M = len(self.tracks)
    self.frame_id += 1
    if N > 0:
        '''Detections'''
        detections = [STrack(STrack.tlbr_to_tlwh(x['bbox']), x['score']) for
                      x in results]
    else:
        detections = []
        
    STrack.multi_predict(self.tracks)
    dist = iou_distance(self.tracks, detections)
    matches, unmatched_tracks , unmatched_dets = linear_assignment(dist, thresh=0.99)

    ret = []
    for itrack, idet in matches:
      tracks=self.tracks[itrack]
      dets = detections[idet]
      tracks.update(dets, self.frame_id)
      ret.append(tracks)
    
    # Private detection: 
    for i in unmatched_dets:
      track = detections[i]
      if track.score > 0.4:
        track.activate(self.kalman_filter, self.frame_id)
        ret.append(track)
        
    self.tracks = ret
    return ret 
