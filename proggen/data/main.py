#!/usr/bin/env python
# coding=utf-8

import os, os.path as osp, sys
from .perception_hard import percept_hard, decode_hdf5_to_frames

def get_dataset_name(name, split):
    if name == 'parabola':
        if split == 'train':
            self.data_fn = osp.join(curdir, 'downloaded_datasets', 'parabola_30K.hdf5')
        elif split == 'test':
            self.data_fn = osp.join(curdir, 'downloaded_datasets', 'parabola_eval.hdf5')
        else:
            raise ValueError(f'Unknown split: {split} for dataset: {name}')
    elif name == 'uniform_motion':
        if split == 'train':
            self.data_fn = osp.join(curdir, 'downloaded_datasets', 'uniform_motion_30K.hdf5')
        elif split == 'test':
            self.data_fn = osp.join(curdir, 'downloaded_datasets', 'uniform_motion_eval.hdf5')
        else:
            raise ValueError(f'Unknown split: {split} for dataset: {name}')
    elif name == 'collision':
        if split == 'train':
            self.data_fn = osp.join(curdir, 'downloaded_datasets', 'collision_30K.hdf5')
        elif split == 'test':
            self.data_fn = osp.join(curdir, 'downloaded_datasets', 'collision_eval.hdf5')
        else:
            raise ValueError(f'Unknown split: {split} for dataset: {name}')
    else:
        raise ValueError(f'Unknown dataset: {name}')
class Dataset:
    def __init__(self, name, split, seed=0):
        curdir = osp.dirname(os.path.abspath(__file__))
        name = get_dataset_name(name, split)
        self.data_fn = osp.join(curdir, 'downloaded_datasets', f'{name}.hdf5')
        assert osp.exists(self.data_fn)
        self.percepted_data_fn = osp.join(curdir, 'percepted_hard_datasets', f'{name}_percepted')
        os.makedirs(self.percepted_data_fn, exist_ok=True)

        data_f = h5py.File(self.data_fn, 'r')
        split_indexes = data_f['video_streams'].keys()
        indexes = [(si, ti) for si in split_indexes for ti in range(len(data_f['video_streams'][si]))]
        rng = np.random.RandomState(seed)
        self.indexes = rng.permutation(indexes)
        data_f.close()
    def __len__(self):
        return len(self.indexes)
    def __getitem__(self, i):
        si, ti = self.indexes[i]
        if osp.exists(self.percepted_data_fn + f'_{si}_{ti}.json'):
            with open(self.percepted_data_fn + f'_{si}_{ti}.json', 'r') as f:
                percepted_data = json.load(f)
        else:
            percept_hard(osp.basename(self.data_fn), si, ti,)
            with open(self.percepted_data_fn + f'_{si}_{ti}.json', 'r') as f:
                percepted_data = json.load(f)

        obj_list = {c for sps in percepted_data['shapes'] for c in sps}
        obj2name = {obj: f'shape{i}' for i, obj in enumerate(sorted(obj_list))}
        trajs = []
        for sps in percepted_data['shapes'][:percepted_data['bad_index_start']]:
            state = {obj2name[obj]: {
                'position': s['position'],
                'angle': 0,
                'shape': 'circle',
                'radius': s['radius'],
                'velocity': None,
                'angular_velocity': None,
            } for obj, s in sps.items()}
            trajs.append(DotMap(state, _dynamic=False))

        gt_positions = percepted_data['gt_feats'][:percepted_data['bad_index_start']]
        assert len(gt_positions) == len(trajs), f'{len(gt_positions)} != {len(trajs)}'
        return {
            'trajs': trajs,
            'gt_positions': gt_positions,
        }
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]