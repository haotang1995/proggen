#!/usr/bin/env python
# coding=utf-8

import os, os.path as osp
import numpy as np
import h5py

from .phyworldphyre import PhyworldPhyreGym, PhyworldPhyreWrapper

def is_iterable(obj):
    try:
        iter(obj)
    except TypeError:
        return False
    return True
class Dataset:
    def __init__(self, seed=0, novideo=False,):
        curdir = osp.dirname(os.path.abspath(__file__))
        name = 'combinatorial_out_of_template_eval_1K'
        self.data_fn = osp.join(curdir, 'downloaded_datasets', f'{name}.hdf5')
        assert osp.exists(self.data_fn)

        data_f = h5py.File(self.data_fn, 'r')
        split_indexes = data_f['video_streams'].keys()
        indexes = [(si, ti) for si in split_indexes for ti in range(len(data_f['video_streams'][si][0]))]
        rng = np.random.RandomState(seed)
        self.indexes = [indexes[i] for i in rng.permutation(len(indexes))]
        data_f.close()

        self.novideo = novideo

        self.env = PhyworldPhyreWrapper(PhyworldPhyreGym(render_mode='rgb_array'))
    def __len__(self):
        return len(self.indexes)
    def __getitem__(self, i):
        if is_iterable(i):
            return [self[j] for j in i]
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        si, ti = self.indexes[i]
        obs, info = self.env.reset(options={'data_split': 'eval', 'trial_index': si})
        trajs = [obs]
        frames = [self.env.render()]
        while True:
            obs, _, done, _, _ = self.env.step(0)
            trajs.append(obs)
            if not self.novideo:
                frames.append(self.env.render())
            if done:
                break

        return {
            'trajs': trajs,
            'frames': frames if not self.novideo else None,
        }
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
