#!/usr/bin/env python
# coding=utf-8

import os, os.path as osp, sys
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))

import time
import numpy as np
import cv2

from proggen.data import Dataset
from proggen.utils.render import OpenCVRender

def test_data():
    dataset = Dataset()
    data = dataset[10]
    print(data)

    video = data['frames'] # numpy array of shape (T, H, W, C)
    trajs = data['trajs'] # list of trajectories
    renderer = OpenCVRender(512, 512, 6)
    # Play the video
    for fi, frame in enumerate(video):
        img = renderer.render(trajs[fi])
        frame = np.concatenate([frame, img], axis=1)
        cv2.imshow('frame', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
        time.sleep(0.2)
    renderer.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test_data()



