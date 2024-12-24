#!/usr/bin/env python
# coding=utf-8

import os, os.path as osp

from typing import Optional
import numpy as np
from dotmap import DotMap
import h5py
import imageio as iio
import tempfile
import imageio

import gymnasium as gym
from gymnasium.utils import EzPickle
from gymnasium import ObservationWrapper
from gymnasium import error, spaces

from .from_phyre.shapes import SCENE_WIDTH as _SCENE_WIDTH, SCENE_HEIGHT as _SCENE_HEIGHT
from ._feature2shape import feature2shape
assert _SCENE_WIDTH == _SCENE_HEIGHT, f"Screens should be square, but got {_SCENE_WIDTH}x{_SCENE_HEIGHT}"

# Needs python 3.9
def decode_hdf5_to_frames_wo_disk(hdf5_path, trial_index, format_hint='.mp4'):
    """Decode video frames from a byte stream stored in an HDF5 file directly in memory."""
    hdf = h5py.File(hdf5_path, 'r')

    byte_stream = hdf['video_streams'][trial_index][0]
    byte_obj = byte_stream.tobytes()
    # Decode frames directly from byte stream
    frames = iio.imread(byte_obj, index=None, extension=format_hint)
    return frames

def decode_hdf5_to_frames(hdf5_path, trial_index):
    """Decode video frames from a byte stream stored in an HDF5 file by first writing to a temporary file."""

    hdf = h5py.File(hdf5_path, 'r')

    byte_stream = hdf['video_streams'][trial_index][0]
    byte_obj = byte_stream.tobytes()
    # Use a temporary file to write the byte stream
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file_name = temp_file.name
        temp_file.write(byte_obj)

    # Now read the video from the temporary file
    with imageio.get_reader(temp_file_name, format='mp4') as reader:
        frames = [frame for frame in reader]
        fps = reader.get_meta_data()['fps']

    # imageio.mimsave('deocded_v1.mp4', frames, fps=fps)

    return np.array(frames), fps


# from gymnasium.error import DependencyNotInstalled
# from gymnasium.utils.step_api_compatibility import step_api_compatibility

class PhyworldPhyreGym(gym.Env, EzPickle):
    SCREEN_WIDTH = 512
    SCREEN_HEIGHT = 512
    FPS = 5
    PPM = 6
    WORLD_SCALE = SCREEN_WIDTH / PPM

    DATA_SPLIT2PATH = {
        'eval': osp.abspath(osp.join(osp.dirname(__file__), 'downloaded_datasets', 'combinatorial_out_of_template_eval_1K.hdf5')),
    }

    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": FPS,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
    ):
        EzPickle.__init__(
            self,
            render_mode,
        )
        self.observation_space = spaces.Box(np.array([-np.inf] * 1000), np.array([np.inf] * 1000), dtype=np.float32)
        self.action_space = spaces.Discrete(1) # Discrete(1) means no action is needed
        self.render_mode = render_mode

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        if options is None: options = {}
        data_split, trial_index = options.get('data_split', 'eval'), options.get('trial_index', None)
        self.data_f = h5py.File(self.DATA_SPLIT2PATH[data_split], 'r')
        if trial_index is None:
            rng = np.random.RandomState(seed)
            trial_index = rng.choice(list(self.data_f['video_streams'].keys()))
        self.trial_index = trial_index
        self.features = np.array(self.data_f['object_streams'][trial_index][0])
        if self.render_mode is not None:
            self.frames, fps = decode_hdf5_to_frames(self.DATA_SPLIT2PATH[data_split], trial_index)
            assert len(self.features) == len(self.frames), f"len(self.features)={len(self.features)}, len(self.frames)={len(self.frames)}"
        self.cur_step = 0
        return feat2obs(self.features[self.cur_step]), {}

    def step(self, action):
        self.cur_step += 1
        return feat2obs(self.features[self.cur_step]), 0, self.cur_step >= len(self.features)-1, False, {}

    def render(self):
        if self.render_mode is None:
            return
        if self.render_mode == "rgb_array":
            return self.frames[self.cur_step]

def feat2obs(feat):
    assert len(feat.shape) == 2
    assert feat.shape[-1] == 14
    shape_num = feat.shape[0]
    return np.concatenate([[shape_num], feat.reshape(-1), [0]*(1000-1-shape_num*14)])
def obs2feat(obs):
    assert len(obs.shape) == 1
    shape_num = int(obs[0])
    assert abs(obs[0] - shape_num) < 1e-6
    return obs[1:1+shape_num*14].reshape(shape_num, 14)

class PhyworldPhyreWrapper(ObservationWrapper):
    def __init__(self, env,):
        super().__init__(env)
        self.env = env
        self.observation_space = None
    def observation(self, obs):
        features = obs2feat(obs)
        out = {
            f'shape_{fi}_{si}': sp
            for fi, feat in enumerate(features)
            for si, sp in enumerate(feature2shape(feat, self.env.unwrapped.SCREEN_WIDTH))
        }
        SCREEN_WIDTH, SCREEN_HEIGHT = self.env.unwrapped.SCREEN_WIDTH, self.env.unwrapped.SCREEN_HEIGHT
        out[f'shape_{len(features)}_0'] = {
            'shape': 'polygon',
            'color': 'white',
            'position': (0, 0),
            'angle': 0,
            'vertices': [
                (0, 0),
                (SCREEN_WIDTH, 0),
                (SCREEN_WIDTH, -10),
                (0, -10),
            ],
        }
        out[f'shape_{len(features)}_1'] = {
            'shape': 'polygon',
            'color': 'white',
            'position': (0, 0),
            'angle': 0,
            'vertices': [
                (0, SCREEN_HEIGHT),
                (SCREEN_WIDTH, SCREEN_HEIGHT),
                (SCREEN_WIDTH, SCREEN_HEIGHT+10),
                (0, SCREEN_HEIGHT+10),
            ],
        }
        out[f'shape_{len(features)}_2'] = {
            'shape': 'polygon',
            'color': 'white',
            'position': (0, 0),
            'angle': 0,
            'vertices': [
                (0, 0),
                (0, SCREEN_HEIGHT),
                (-10, SCREEN_HEIGHT),
                (-10, 0),
            ],
        }
        out[f'shape_{len(features)}_3'] = {
            'shape': 'polygon',
            'color': 'white',
            'position': (0, 0),
            'angle': 0,
            'vertices': [
                (SCREEN_WIDTH, 0),
                (SCREEN_WIDTH, SCREEN_HEIGHT),
                (SCREEN_WIDTH+10, SCREEN_HEIGHT),
                (SCREEN_WIDTH+10, 0),
            ],
        }
        for shape in out.values():
            # shape['velocity'] = (0., 0.) if self.env.unwrapped.cur_step == 0 else 'NULL'
            # shape['angular_velocity'] = 0. if self.env.unwrapped.cur_step == 0 else 'NULL'
            shape['position'] = tuple(p/self.env.unwrapped.PPM for p in shape['position'])
            if 'vertices' in shape:
                shape['vertices'] = [tuple(p/self.env.unwrapped.PPM for p in v) for v in shape['vertices']]
            else:
                assert 'radius' in shape
                shape['radius'] /= self.env.unwrapped.PPM
        return DotMap(out)

gym.register(
    id = "PhyworldPhyre-v0",
    entry_point = PhyworldPhyreGym,
    max_episode_steps=50,
    reward_threshold=1,
    kwargs={
    },
)
