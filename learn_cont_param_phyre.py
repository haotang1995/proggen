#!/usr/bin/env python
# coding=utf-8

import os, os.path as osp
import itertools
import json
from pprint import pprint
from dotmap import DotMap

import numpy as np
import matplotlib.pyplot as plt
import argparse

from proggen.prog import FixtureDef, BodyDef, JointDef, ContactableGraph
from proggen.prog import MetaBox2DProgram, ContParams, Box2DProgram
from proggen.prog.prog import _metrics_fn
from proggen.utils.logger import set_logger
from proggen.utils.render import PygameRender
from proggen.utils.render import OpenCVPHYRERender
from proggen.data import Dataset

def inv_sigmoid(x):
    return np.log(x/(1-x))
def to_serializable(obj):
    if isinstance(obj, DotMap):
        return to_serializable(obj.toDict())
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.bool_)):
        return bool(obj)
    if isinstance(obj, (list, tuple)):
        return [to_serializable(o) for o in obj]
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    return obj
def prog_fn(state):
    fixture_names = set(state.keys())
    fixtures = {
        fn: FixtureDef(fn, state[fn]['shape'],)
        for fn in fixture_names
    }
    body_names = set(k[:k.rfind('_')] for k in fixtures)
    body_def = {bn: BodyDef(
        bn, [k for k in fixtures if k.startswith(bn+'_')],
        'dynamic' if state[bn+'_0']['color'].lower() in ['gray', 'green', 'blue', 'red'] else 'static'
    ) for bn in body_names}
    # print('body:', {bn: body.body_type for bn, body in body_def.items()})
    joint_def = dict()
    # print('joint:', {jn: joint.joint_type for jn, joint in joint_def.items()})

    program = Box2DProgram(fixtures, body_def, joint_def)
    return program
def get_program():
    return MetaBox2DProgram(prog_fn)
def predict(trajs_list, program, params, free_init=False):
    pred_trajs_list = []
    for ti, trajs in enumerate(trajs_list):
        if free_init:
            pred_trajs, params, initial_state_params = program._simulate_from_partial_trajs(params, trajs[:4], FPS, len(trajs)-1,)
        else:
            initial_state_params = program._get_initial_state_params(trajs[:4], FPS,)
            pred_trajs, params, initial_state_params = program._simulate(params, initial_state_params, trajs[0], FPS, len(trajs)-1,)
        assert len(pred_trajs) == len(trajs)
        pred_trajs_list.append(pred_trajs)
    return pred_trajs_list, params
def evaluate(pred_trajs_list, trajs_list, frames_list, verbose=False):
    assert False
    if verbose:
        if len(trajs_list[0][0]) == 1:
            plot_trajs_diff(trajs_list[0], pred_trajs_list[0], gt_positions_list[0])
        elif len(trajs_list[0][0]) == 2:
            plot_trajs_diff_collision(trajs_list[0], pred_trajs_list[0], gt_positions_list[0])
        else:
            raise ValueError(f'unsupported object size: {len(trajs_list[0][0])}')
    metrics, vel_err = zip(*[_metrics_fn(tpt, ot, fps=FPS, gt_positions=ogt) for tpt, ot, ogt in zip(pred_trajs_list, trajs_list, gt_positions_list)])
    metrics = [{k: np.mean([v[k] for v in mm.values()]) for k in list(mm.values())[0].keys()} for mm in metrics]
    vel_err = list(vel_err)
    metrics = {k: np.mean([mm[k] for mm in metrics]) for k in metrics[0].keys()}
    vel_err = {k: np.mean([v[k] for v in vel_err]) for k in vel_err[0].keys()}
    if verbose:
        pprint(metrics)
        pprint(vel_err)
    return {
        'metrics': metrics,
        'vel_err': vel_err,
    }
def save_video(trajs, frames, outdir, outname):
    render = OpenCVPHYRERender(512, 512, 6)
    frames, comp_frames = [], []
    for ti, s in enumerate(trajs):
        img = render.render(s)
        frames.append(img)
        comp_frames.append(np.concatenate([img, frames[ti]], axis=1))
    render.close()
    # Save video as mp4 (numpy frames, [T, H, W, C])
    outname = osp.join(outdir, outname) + '.mp4'
    comp_outname = outname.replace('.mp4', '_comp.mp4')
    render.save_video(frames, outname)
    render.save_video(comp_frames, comp_outname)

FPS = 10
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--free-init', action='store_true', default=False)

    parser.add_argument('--loss_name', type=str, default='mae')
    parser.add_argument('--early_stop_threshold', type=float, default=0.1)
    parser.add_argument('--optimizer', type=str, default='Powell')
    # parser.add_argument('--maxiter', type=int, default=int(1e7))
    parser.add_argument('--maxiter', type=int, default=int(10))
    parser.add_argument('--max_tries', type=int, default=1)

    args = parser.parse_args()

    train_dataset = Dataset(seed=0, novideo=True,)

    optim_hyperparams = {
        'loss_name': args.loss_name,
        'early_stop_threshold': args.early_stop_threshold,
        'optimizer': args.optimizer,
        'maxiter': args.maxiter,
        'max_tries': args.max_tries,
    }

    curdir = osp.dirname(os.path.abspath(__file__))
    outdir = osp.join(curdir, 'results', osp.basename(__file__).replace('.py', ''))
    os.makedirs(outdir, exist_ok=True)
    outname = osp.join(outdir, f'bs{args.batch_size}' + ('_freeinit' if args.free_init else '') + '.json')
    video_outdir = outname[:-5] + '_videos'
    os.makedirs(video_outdir, exist_ok=True)

    train_data_list = [train_dataset[i] for i in range(args.batch_size)]
    trajs_list = [data['trajs'] for data in train_data_list]
    frames_list = [data['frames'] for data in train_data_list]

    program = get_program()

    set_logger(outname.replace('.json', '.log'))
    set_params = {
        'gravity': (0., -9.8),
        'fixture_density': (0.25,),
        'fixture_friction': (inv_sigmoid(0.5),),
        'fixture_restitution': (inv_sigmoid(0.2),),
    }
    set_params = {}
    if args.free_init:
        params = program.fit_all(trajs_list, FPS, verbose=True, set_params=set_params, hyperparams_list=[optim_hyperparams,], batched=True,)
    else:
        params = program.fit(trajs_list, FPS, verbose=True, set_params=set_params, hyperparams_list=[optim_hyperparams,], batched=True,)
    pred_trajs_list, params = predict(trajs_list, program, params, args.free_init)
    params.set_params(set_params)
    pred_trajs_list, params = predict(trajs_list, program, params, args.free_init)
    pprint(params)
    pred_frames_list = []
    for ti, pred_trajs in enumerate(pred_trajs_list):
        pred_frames = save_video(pred_trajs, frames_list[ti], video_outdir, f'train_{ti:03d}')
        pred_frames_list.append(pred_frames)
    assert False
    train_metrics = evaluate(pred_frames_list, frames_list, verbose=False,)
    pprint(train_metrics)

    test_data_list = [train_dataset[-(i+1)] for i in range(0, 10)] # test on the last 10 data, just for now
    print(f'Test data size: {len(test_data_list)}')
    trajs_list = [data['trajs'] for data in test_data_list]
    frames_list = [data['frames'] for data in test_data_list]
    pred_trajs_list, _ = predict(trajs_list, program, params, args.free_init)
    test_metrics = evaluate(pred_trajs_list, trajs_list, frames_list, verbose=False,)
    pprint(test_metrics)
    with open(outname, 'w') as f:
        json.dump({
            'params': params.dumps(),
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
        }, f, indent=2)

if __name__ == '__main__':
    main()
