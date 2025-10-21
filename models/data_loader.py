from __future__ import division

import numpy as np
from torch.utils.data import Dataset


def vidstart2trunks(vid_start, seqlen):
    vid_start_trunks = []
    vid_length_trunks = []
    vid_id_trunks = []
    N = len(vid_start)
    for n in range(N-1):
        for i in range(vid_start[n], vid_start[n+1]):
            length = min(seqlen, vid_start[n+1]-i)
            vid_start_trunks.append(i)
            vid_length_trunks.append(length)
            vid_id_trunks.append(n)

    return (vid_start_trunks, vid_length_trunks, vid_id_trunks)

class video_test(Dataset):
    """
    Base Dataset Class - Handles data loading and augmentation.
    Able to handle heterogeneous datasets (different annotations available for different datasets).
    You need to update the path to each dataset in utils/config.py.
    """
    def __init__(self, pred_verts, seqlen):
        super(video_test, self).__init__()

        self.pred_verts = pred_verts
        self.seqlen = seqlen

        self.vid_start, self.vid_length, self.vid_id = vidstart2trunks([0, len(self.pred_verts)], seqlen)
        self.len = len(self.vid_start)

    def __getitem__(self, index):

        vid_start = self.vid_start[index]
        length = self.vid_length[index]

        # make Y
        Y = {}

        Y['pred_verts']  = np.zeros([self.seqlen, 778, 3]).astype('float')
        Y['pred_verts'][:length] = self.pred_verts[vid_start:vid_start+length]

        Y['src_mask'] = np.zeros((self.seqlen), dtype=bool)
        Y['src_mask'][:length] = True     

        return Y

    def __len__(self):
        return self.len