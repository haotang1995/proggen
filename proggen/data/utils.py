#!/usr/bin/env python
# coding=utf-8

import numpy as np
import h5py
import tempfile
import imageio

# Copied from Phyworld:
# https://github.com/phyworld/phyworld/blob/1c65f0f8dc7a2d9fb4e018767819f91a2980cd3e/combinatorial_data/data_generator_v2.py#L144
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

    return np.array(frames), fps
