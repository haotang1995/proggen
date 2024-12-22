#!/usr/bin/env python
# coding=utf-8

import os, os.path as osp, sys
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..', '..')))

import copy
import itertools
from pprint import pprint

import random
import numpy as np
import gymnasium as gym

from Box2D import b2World

from cont_world_coder.envs.render import PygameRender, OpenCVRender
from cont_world_coder.envs.lunar_lander.lunar_lander import LunarLanderWrapper
from cont_world_coder.envs.cartpole.cartpole import CartPoleWrapper
from cont_world_coder.envs.phyworld.two_ball_collision import PhyworldTwoBallCollisionWrapper
from cont_world_coder.envs.phyworld.one_ball_uniform_motion import PhyworldOneBallUniformMotionWrapper
from cont_world_coder.envs.phyworld.one_ball_parabola import PhyworldOneBallParabolaWrapper
from cont_world_coder.envs.phyworld_phyre.phyworldphyre import PhyworldPhyreWrapper

from cont_world_coder.dynamics.llm2prog import LLM2Prog, template_OriginalScreenshot_AnnotatedContour_WithoutSemanticLabels
from cont_world_coder.dynamics.llm2prog import template_OriginalScreenshot_AnnotatedContour_WithoutSemanticLabels_StackVideo
from cont_world_coder.dynamics.dsl import code2prog

def test_llm2prog_cartpole():
    seed = 2
    random.seed(seed)
    np.random.seed(seed)
    # env = CartPoleWrapper(gym.make('CartPole-v1', render_mode='rgb_array',))
    # env = LunarLanderWrapper(gym.make('LunarLander-v3', render_mode='rgb_array',))
    # env = PhyworldTwoBallCollisionWrapper(gym.make('PhyworldTwoBallCollision-v0', render_mode='rgb_array',))
    # env = PhyworldOneBallUniformMotionWrapper(gym.make('PhyworldOneBallUniformMotion-v0', render_mode='rgb_array',))
    # env = PhyworldOneBallParabolaWrapper(gym.make('PhyworldOneBallParabola-v0', render_mode='rgb_array',))
    env = PhyworldPhyreWrapper(gym.make('PhyworldPhyre-v0', render_mode='rgb_array',))
    render = OpenCVRender(env.unwrapped.SCREEN_WIDTH, env.unwrapped.SCREEN_HEIGHT, env.unwrapped.PPM)

    obs, info = env.reset(seed=0)
    trajs = []
    screenshots = []
    for t in range(100):
        action = 1 if isinstance(env, CartPoleWrapper) else 0 # Do thing action
        obs, reward, terminated, truncated, info = env.step(action)
        trajs.append(obs)
        screenshots.append(env.render())
        # screenshots.append(render.render(obs))
        # import matplotlib.pyplot as plt
        # plt.imshow(screenshots[-1])
        # plt.show()
        if terminated or truncated:
            break
    env.close()

    screenshots = np.array(screenshots)

    # llm2prog = LLM2Prog(template_OriginalScreenshot_AnnotatedContour_WithoutSemanticLabels)
    llm2prog = LLM2Prog(template_OriginalScreenshot_AnnotatedContour_WithoutSemanticLabels_StackVideo)
    code = llm2prog(trajs, screenshots, env.unwrapped.PPM)
    # code = llm2prog(trajs, screenshots, env.unwrapped.PPM)
    # code = llm2prog(trajs, screenshots, env.unwrapped.PPM)
    print(code)

if __name__ == '__main__':
    test_llm2prog_cartpole()


