#!/usr/bin/env python
# coding=utf-8

import os, os.path as osp, sys
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))

import copy
import itertools
import json
from pprint import pprint
import time
import h5py
from dotmap import DotMap

import random
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import argparse

from prog_def import FixtureDef, BodyDef, JointDef, ContactableGraph
from prog import Box2DProgram, ContParams, _metrics_fn, _decode_loss
from logger import set_logger
from _render import PygameRender
from perception_hard import percept_hard

def render_trajs(trajs):
    render = PygameRender(FPS, 256, 256, 256/10.)
    for ti, s in enumerate(trajs):
        render.render(s)
    render.close()
def plot_trajs_diff(trajs, pred_trajs, gt_positions):
    H, W = 2, 2
    fig, axes = plt.subplots(H, W, figsize=(H*5, W*5))
    mae0, mae1 = 0, 0
    pred_vel_err_avg, gt_vel_err_avg = 0, 0
    pred_vel_error, gt_vel_error = 0, 0
    for oi, obj in enumerate(sorted(trajs[0])):
        gt = np.array([s[obj]['position'] for s in trajs])
        pred = np.array([s[obj]['position'] for s in pred_trajs])
        gt_pos = np.array([s for s in gt_positions])
        axes[0,0].plot(gt[:, 0], gt[:, 1], 'rx-')
        axes[0,0].plot(pred[:, 0], pred[:, 1], 'bx-')
        axes[0,0].plot(gt_pos[:, 0], gt_pos[:, 1], 'gx-')
        mae0 += np.mean(np.abs(gt[1:,0] - pred[1:,0]))
        mae1 += np.mean(np.abs(gt[1:,1] - pred[1:,1]))

        axes[0,1].plot(gt[:,0], 'rx-')
        axes[0,1].plot(pred[:,0], 'bx-')
        axes[0,1].plot(gt_pos[:,0], 'gx-')

        gt_vel = np.diff(gt, axis=0) * FPS
        pred_vel = np.diff(pred, axis=0) * FPS
        gt_pos_vel = np.diff(gt_pos, axis=0) * FPS
        axes[1,0].plot(gt_vel[:,0], 'rx-')
        axes[1,0].plot(pred_vel[:,0], 'bx-')
        axes[1,0].plot(gt_pos_vel[:,0], 'gx-')
        gt_vel_err = np.abs(gt_pos_vel[:,0] - gt_vel[:,0])
        pred_vel_err = np.abs(pred_vel[:,0] - gt_vel[:,0])
        axes[1,1].plot(gt_vel_err, 'rx-')
        axes[1,1].plot(pred_vel_err, 'bx-')
        axes[1,1].set_ylim(0, 0.5)
        pred_vel_err_avg += np.mean(np.abs(pred_vel[4:,0] - gt_pos_vel[4:,0]))
        gt_vel_err_avg += np.mean(np.abs(gt_vel[4:,0] - gt_pos_vel[4:,0]))
        pred_vel_error += np.abs(pred_vel[4:,0].mean() - gt_pos_vel[4:,0].mean())
        gt_vel_error += np.abs(gt_vel[4:,0].mean() - gt_pos_vel[4:,0].mean())
    mae0 /= len(trajs[0])
    mae1 /= len(trajs[0])
    mae = (mae0 + mae1) / 2
    axes[0,0].set_title(f'mae: {mae:.4f}; mae0: {mae0:.4f}; mae1: {mae1:.4f}')
    axes[0,0].set_xlabel('x')
    axes[0,0].set_ylabel('y')
    axes[0,1].set_xlabel('t')
    axes[0,1].set_ylabel('x')
    axes[1,0].set_xlabel('t')
    axes[1,0].set_ylabel('vx')
    pred_vel_err_avg /= len(trajs[0])
    gt_vel_err_avg /= len(trajs[0])
    pred_vel_error /= len(trajs[0])
    gt_vel_error /= len(trajs[0])
    axes[1,1].set_title(f'pred_vel_err_avg: {pred_vel_err_avg:.4f}, gt_vel_err_avg: {gt_vel_err_avg:.4f}\n'
                        f'pred_vel_error: {pred_vel_error:.4f}, gt_vel_error: {gt_vel_error:.4f}')
    axes[1,1].set_xlabel('t')
    axes[1,1].set_ylabel('vx_err')
    plt.show()
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
def fit(trajs_list, other_trajs, gt_positions_list, other_gt_positions, outname, hyperparams, oracle_nofit=False, bounded_friction_restitution=False,):
    set_logger(outname.replace('.json', '.log'))
    fixture_names = set(k for obs in trajs_list[0] for k in obs)
    fixtures = {
        fn: FixtureDef(fn, trajs_list[0][0][fn]['shape'], bounded_friction_restitution=bounded_friction_restitution,)
        for fn in fixture_names
    }
    contactable_graph = ContactableGraph({tuple(sorted([k1, k2])): True for k1, k2 in itertools.product(fixtures.keys(), repeat=2)})
    print('contactable graph:', {tuple(sorted([k1, k2])) for k1, k2 in itertools.combinations(fixtures.keys(), 2) if contactable_graph.query_category(k1) & contactable_graph.query_mask(k2) and contactable_graph.query_category(k2) & contactable_graph.query_mask(k1)})
    body_def = {'ball1': BodyDef('ball1', ['shape0'], 'dynamic'),}
    print('body:', {bn: body.body_type for bn, body in body_def.items()})
    joint_def = dict()
    print('joint:', {jn: joint.joint_type for jn, joint in joint_def.items()})
    program = Box2DProgram(fixtures, contactable_graph, body_def, joint_def)

    fit_params = True
    # fit_params = False
    # nofit = True
    nofit = True
    # oracle_nofit=False
    # oracle_nofit=True
    nofit = nofit or oracle_nofit
    # hyperparams['maxiter'] = 1
    if fit_params:
        if nofit:
            params = program.fit(trajs_list, FPS, verbose=True, set_params={}, hyperparams_list=[hyperparams,], nofit=nofit, oracle_nofit=oracle_nofit, batched=True,)
            initial_state_params_list = [program._get_initial_state_params(trajs, FPS, oracle_nofit=oracle_nofit,) for trajs in trajs_list]
        else:
            params = program.fit_all(trajs_list, FPS, verbose=True, set_params={}, hyperparams_list=[hyperparams,], batched=True,)
            initial_state_params_list = [params[i*params.shape[0]//(len(trajs_list) + 1):(i+1)*params.shape[0]//(len(trajs_list) + 1)] for i in range(len(trajs_list))]
            params = params[:params.shape[0]//(len(trajs_list) + 1)]
            params = ContParams(params)
            initial_state_params_list = [ContParams(initial_state_params) for initial_state_params in initial_state_params_list]

        # params = program.fit([trajs] + other_trajs[:3], FPS, verbose=True, set_params={},
                            # hyperparams_list=[hyperparams,], batched=True)
        # other_trajs = other_trajs[3:]
        # other_gt_positions = other_gt_positions[3:]

    else:
        params = ContParams(np.random.rand(1000))
        params.insert_param('gravity', (0., 0.))
        params.insert_param('shape0_fixture_density', (1.0,))
        params.insert_param('shape0_fixture_friction', (0.2,))
        params.insert_param('shape0_fixture_restitution', (1.,))
        params.insert_param('shape1_fixture_density', (1.0,))
        params.insert_param('shape1_fixture_friction', (0.2,))
        params.insert_param('shape1_fixture_restitution', (1.,))
        if nofit:
            initial_state_params = [program._get_initial_state_params(trajs, FPS, oracle_nofit=oracle_nofit) for trajs in trajs_list]
        else:
            initial_state_params = []
            for trajs in trajs_list:
                pred_trajs, params, initial_state_params = program._simulate_from_partial_trajs(params, trajs, FPS, len(trajs)-1,)
                initial_state_params_list.append(initial_state_params)

    train_pred_trajs = []
    for ti, trajs in enumerate(trajs_list):
        pred_trajs, params, initial_state_params = program._simulate(params, initial_state_params_list[ti], trajs[0], FPS, len(trajs)-1,)
        train_pred_trajs.append(pred_trajs)
        initial_state_params_list[ti] = initial_state_params
    # train_pred_trajs = [program._simulate(params, initial_state_params, trajs[0], FPS, len(trajs)-1,) for trajs, initial_state_params in zip(trajs_list, initial_state_params_list)]
    print(params)
    print(initial_state_params_list)
    train_metrics, train_vel_err = zip(*[_metrics_fn(tpt, ot, fps=FPS, gt_positions=ogt) for tpt, ot, ogt in zip(train_pred_trajs, trajs_list, gt_positions_list)])
    train_metrics = [{k: np.mean([v[k] for v in mm.values()]) for k in list(mm.values())[0].keys()} for mm in train_metrics]
    train_vel_err = list(train_vel_err)
    pprint({k: np.mean([mm[k] for mm in train_metrics]) for k in train_metrics[0].keys()})
    pprint({k: np.mean([v[k] for v in train_vel_err]) for k in train_vel_err[0].keys()})
    print([ve['percepted_pred_vel_error'] for ve in train_vel_err], np.mean([ve['percepted_pred_vel_error'] for ve in train_vel_err]))
    plot_trajs_diff(trajs_list[0], train_pred_trajs[0], gt_positions_list[0])
    # assert False
    test_pred_trajs = [program._simulate_from_partial_trajs(params, ot, FPS, len(ot)-1, nofit=nofit, oracle_nofit=oracle_nofit)[0] for ot in other_trajs]
    test_metrics = [_metrics_fn(tpt, ot, fps=FPS, gt_positions=ogt) for tpt, ot, ogt in zip(test_pred_trajs, other_trajs, other_gt_positions)]
    test_metrics, test_vel_err = zip(*test_metrics)
    test_metrics = [{k: np.mean([v[k] for v in mm.values()]) for k in list(mm.values())[0].keys()} for mm in test_metrics]
    test_vel_err = list(test_vel_err)
    pprint({k: np.mean([v[k] for v in test_vel_err]) for k in test_vel_err[0].keys()})
    assert len(test_vel_err) == 9, len(test_vel_err)
    print([ve['percepted_pred_vel_error'] for ve in test_vel_err], np.mean([ve['percepted_pred_vel_error'] for ve in test_vel_err]))

    with open(outname, 'w') as f:
        json.dump({
            'params': params.dumps(),
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'train_trajs': to_serializable(trajs),
            'other_trajs': to_serializable(other_trajs),
        }, f, indent=2)

    assert False

FPS = 10
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--oracle_nofit', action='store_true')
    parser.add_argument('--bound_fix_res', action='store_true')
    args = parser.parse_args()

    curdir = osp.dirname(os.path.abspath(__file__))
    data_fn = osp.join(curdir, 'downloaded_datasets', 'parabola_30K.hdf5')
    assert osp.exists(data_fn)
    data_f = h5py.File(data_fn, 'r')
    split_indexes = data_f['video_streams'].keys()
    indexes = [(si, ti) for si in split_indexes for ti in range(len(data_f['video_streams'][si]))]

    rng = np.random.RandomState(0)
    _indexes = [indexes[i] for i in rng.choice(len(indexes), 10, replace=False)]
    _left_indexes = [i for i in indexes if i not in _indexes]
    batched_indexes = [_left_indexes[j] for j in rng.choice(len(_left_indexes), 100, replace=False)]
    _left_indexes = [i for i in _left_indexes if i not in batched_indexes]
    rng.shuffle(_left_indexes)
    batched_indexes = batched_indexes + _left_indexes
    indexes = _indexes + batched_indexes[:args.batch_size]
    assert len(indexes) == 10 + args.batch_size

    percepted_data_fn = osp.join(curdir, 'percepted_hard_datasets', 'parabola_30K_percepted')
    percepted_data_list = []
    for si, ti in indexes:
        if osp.exists(percepted_data_fn + '.json'):
            with open(percepted_data_fn + '.json', 'r') as f:
                percepted_data = json.load(f)[si][ti]
        elif osp.exists(percepted_data_fn + f'_{si}.json'):
            with open(percepted_data_fn + f'_{si}.json', 'r') as f:
                percepted_data = json.load(f)[ti]
        elif osp.exists(percepted_data_fn + f'_{si}_{ti}.json'):
            with open(percepted_data_fn + f'_{si}_{ti}.json', 'r') as f:
                percepted_data = json.load(f)
        else:
            percept_hard(osp.basename(data_fn), si, ti,)
            with open(percepted_data_fn + f'_{si}_{ti}.json', 'r') as f:
                percepted_data = json.load(f)
            # raise FileNotFoundError(percepted_data_fn + f'_{si}_{ti}.json')
        percepted_data_list.append(percepted_data)

    gt_positions_list = []
    trajs_list = []
    for data in percepted_data_list:
        obj_list = {c for sps in data['shapes'] for c in sps}
        obj2name = {obj: f'shape{i}' for i, obj in enumerate(sorted(obj_list))}
        avg_radius = {obj: np.mean([s[obj]['radius'] for s in data['shapes'][:data['bad_index_start']]]) for obj in obj_list}
        trajs = []
        for sps in data['shapes'][:data['bad_index_start']]:
            state = {obj2name[obj]: {
                'position': s['position'],
                'angle': 0,
                'shape': 'circle',
                'radius': avg_radius[obj],
                'velocity': None,
                'angular_velocity': None,
            } for obj, s in sps.items()}
            trajs.append(DotMap(state, _dynamic=False))
        trajs_list.append(trajs)

        gt_positions = data['gt_feats'][:data['bad_index_start']]
        assert len(gt_positions) == len(trajs), f'{len(gt_positions)} != {len(trajs)}'
        gt_positions_list.append(gt_positions)
    assert len(trajs_list) == 10 + args.batch_size, len(trajs_list)


    hyperparams_list = {
        'mae_powell': {
            'loss_name': 'mae',
            'early_stop_threshold': 0.1,
            'optimizer': 'Powell',
            'maxiter': int(1e7),
            'max_tries': 10,},
    }

    outdir = osp.join(curdir, 'results', osp.basename(__file__).replace('.py', ''))
    os.makedirs(outdir, exist_ok=True)

    for ti in range(0, 10):
        other_trajs = trajs_list[:ti] + trajs_list[ti+1:10]
        trajs_list = [trajs_list[ti],] + trajs_list[10:]
        other_gt_positions = gt_positions_list[:ti] + gt_positions_list[ti+1:]
        gt_positions_list = [gt_positions_list[ti],] + gt_positions_list[10:]
        for hyperparams_name, hyperparams in hyperparams_list.items():
            outname = osp.join(outdir, f'{ti}_{hyperparams_name}.json')
            # if osp.exists(outname):
                # continue
            assert len(other_trajs) == 9, len(other_trajs)
            fit(trajs_list, other_trajs, gt_positions_list, other_gt_positions, outname, hyperparams, oracle_nofit=args.oracle_nofit, bounded_friction_restitution=args.bound_fix_res,)

if __name__ == '__main__':
    main()
