#!/usr/bin/env python
# coding=utf-8

import numpy as np
if __name__ == '__main__':
    from from_phyre.shapes import get_builders, Jar
else:
    from .from_phyre.shapes import get_builders, Jar

# Copied from PHYRE
def _get_jar_offset(diameter):
    center_x, center_y = Jar.center_of_mass(**dict(
        diameter=diameter))
    return center_y

def feature2shape(feature, WORLD_SCALE):
    # assert SCENE_WIDTH == SCENE_HEIGHT
    '''
    0: x in pixels of center of mass divided by SCENE_WIDTH

    1: y in pixels of center of mass divided by SCENE_HEIGHT

    2: angle of the object between 0 and 2pi divided by 2pi

    3: diameter in pixels of object divided by SCENE_WIDTH

    4-8: One hot encoding of the object shape, according to order:
        ball, bar, jar, standing sticks

    8-14: One hot encoding of object color, according to order:
        red, green, blue, purple, gray, black
    '''
    assert feature.shape == (14,)

    color = feature[8:].argmax()
    color = ['red', 'green', 'blue', 'purple', 'gray', 'black'][color]

    shape = feature[4:8].argmax()
    shape = ['ball', 'bar', 'jar', 'standingsticks'][shape]

    x = feature[0] * WORLD_SCALE
    y = feature[1] * WORLD_SCALE
    angle = feature[2] * 2 * np.pi
    diameter = feature[3] * WORLD_SCALE

    # Shape info extracted from PHYRE:
    # https://github.com/facebookresearch/phyre/blob/d0765dd2f6d6cf41b933a3a57173c63d21fbfa6a/src/python/phyre/creator/shapes.py#L338
    if shape == 'ball':
        return [{
            'shape': 'circle',
            'color': color,
            'position': (x, y),
            'angle': angle,
            'radius': diameter / 2,
        }]
    else:
        builder = get_builders()[shape]
        shapes, phantom_vertices = builder.build(diameter=diameter,)
        if shape == 'jar':
            # dx, dy = builder.center_of_mass(diameter=diameter)
            dy = _get_jar_offset(diameter)
            dx = -1 * dy * np.sin(angle)
            dy = dy * np.cos(angle)
        else:
            dx, dy = 0, 0
        return [{
            'shape': 'polygon',
            'color': color,
            'position': (x-dx, y-dy),
            'angle': angle,
            'vertices': [(v.x, v.y) for v in s.polygon.vertices],
        } for s in shapes]

if __name__ == '__main__':
    print('ONLY FOR TESTING')
    import h5py
    from utils import decode_hdf5_to_frames
    import matplotlib.pyplot as plt
    f = h5py.File('./downloaded_datasets/combinatorial_out_of_template_eval_1K.hdf5', 'r')
    trial_index = '10066:016'
    features = np.array(f['object_streams'][trial_index])
    frames, fps = decode_hdf5_to_frames('./downloaded_datasets/combinatorial_out_of_template_eval_1K.hdf5', trial_index)
    WORLD_SCALE = frames[0].shape[0]
    # WORLD_SCALE = 256
    fixtures = []
    for feature in features[0][0]:
        try:
            fixtures.extend(feature2shape(feature, WORLD_SCALE))
        except NotImplementedError:
            pass

    def rotate_vertex(x, y, angle):
        return x * np.cos(angle) - y * np.sin(angle), x * np.sin(angle) + y * np.cos(angle)

    print(features[0][0].shape)
    print(fixtures)
    plt.imshow(frames[0])
    for feature in features[0][0]:
        x, y, angle, diameter = feature[:4]
        x, y = x * WORLD_SCALE, y * WORLD_SCALE
        diameter *= WORLD_SCALE
        angle *= 2 * np.pi
        plt.gca().add_artist(plt.Circle((x, WORLD_SCALE-y), diameter/2, fill=False, edgecolor='black'))
        plt.plot(x, WORLD_SCALE-y, 'rx')
    # plt.show()
    # assert False
    for fixture in fixtures:
        if fixture['shape'] == 'circle':
            x, y = fixture['position']
            r = fixture['radius']
            plt.gca().add_artist(plt.Circle((x, WORLD_SCALE-y), r, fill=False, edgecolor=fixture['color']))
        elif fixture['shape'] == 'polygon':
            x, y = fixture['position']
            vertices = fixture['vertices']
            vertices = [rotate_vertex(vx, vy, fixture['angle']) for vx, vy in vertices]
            vertices = [(x + vx, WORLD_SCALE - (y + vy)) for vx, vy in vertices]
            plt.gca().add_artist(plt.Polygon(vertices, fill=False, edgecolor=fixture['color']))
    plt.show()
