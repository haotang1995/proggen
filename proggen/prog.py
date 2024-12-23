#!/usr/bin/env python
# coding=utf-8

import copy
import itertools
from dotmap import DotMap
import numpy as np
from scipy.optimize import minimize

from Box2D import b2World, b2Vec2, b2Transform
from Box2D import b2TestOverlap, b2PolygonShape, b2CircleShape, b2EdgeShape
from Box2D import b2FixtureDef, b2BodyDef, b2_dynamicBody, b2_staticBody, b2_kinematicBody

class FixtureDef:
    def __init__(self, name, shape_name):
        self.name = name
        self.shape_name = shape_name
        self._kwargs = None
    def init_def(self, params, categoryBits=None, maskBits=None):
        assert (categoryBits is None) == (maskBits is None), "categoryBits and maskBits must be both None or both not None"
        out = copy.deepcopy(self)
        out._kwargs = {
            'density': abs(params.get_params(f'{self.name}_fixture_density')),
            'friction': abs(params.get_params(f'{self.name}_fixture_friction')),
            'restitution': abs(params.get_params(f'{self.name}_fixture_restitution')),
        }
        if categoryBits is not None:
            out._kwargs['categoryBits'] = categoryBits
        if maskBits is not None:
            out._kwargs['maskBits'] = maskBits
        return out
    def add_to_body(self, body, state, transform):
        _kwargs = copy.deepcopy(self._kwargs)
        assert state['shape'] == self.shape_name, (state['shape'], self.shape_name)
        if self.shape_name == 'circle':
            circle = b2CircleShape(radius=state['radius'], )
            circle.position = _exec_transform(transform, (0, 0))
            fixture_def = b2FixtureDef(**_kwargs, shape=circle)
            body.CreateFixture(fixture_def)
        elif self.shape_name == 'polygon':
            polygon = b2PolygonShape(vertices=[_exec_transform(transform, tuple(v)) for v in state['vertices']])
            fixture_def = b2FixtureDef(**_kwargs, shape=polygon)
            body.CreateFixture(fixture_def)
        else:
            vertices = [_exec_transform(transform, tuple(v)) for v in state['vertices']]
            for edge in zip(vertices, vertices[1:] + [vertices[0]]):
                shape = b2EdgeShape(vertices=edge)
                fixture_def = b2FixtureDef(**_kwargs, shape=shape)
                body.CreateFixture(fixture_def)

class ContactableGraph:
    def __init__(self, contactable):
        self.contactable = contactable
        fixture_names = sorted(set([fn for fns in self.contactable.keys() for fn in fns]))
        self.categoryBits = {fn: 1 << i for i, fn in enumerate(fixture_names)}
        self.maskBits = {
            fn: sum([1 << i for i, fn2 in enumerate(fixture_names) if fn != fn2 and self.contactable[tuple(sorted((fn, fn2)))]])
            for fn in fixture_names
        }
    def query_category(self, fixture_name):
        return self.categoryBits[fixture_name]
    def query_mask(self, fixture_name):
        return self.maskBits[fixture_name]

def _exec_transform(transform, v):
    pos_diff = transform['position_diff']
    angle_diff = transform['angle_diff']
    x, y = v
    return (
        x * np.cos(angle_diff) - y * np.sin(angle_diff) + pos_diff[0],
        x * np.sin(angle_diff) + y * np.cos(angle_diff) + pos_diff[1],
    )
def float_vec(vec):
    return (float(v) for v in vec)
BODYTYPE = {
    'dynamic': b2_dynamicBody,
    'static': b2_staticBody,
    'kinematic': b2_kinematicBody,
}
class BodyDef:
    def __init__(self, name, fixture_names, body_type,):
        self.name = name
        self.fixture_names = fixture_names
        self.body_type = body_type.lower()
    def init(self, params, world, fixture_definitions, initial_state):
        self.initial_fixure_positions = {fn: b2Vec2(*float_vec(initial_state[fn]['position'])) for fn in self.fixture_names}
        self.initial_fixture_velocities = {fn: b2Vec2(*float_vec(initial_state[fn]['velocity'])) for fn in self.fixture_names}
        self.initial_fixture_angles = {fn: float(initial_state[fn]['angle']) for fn in self.fixture_names}
        self.initial_fixture_angular_velocities = {fn: float(initial_state[fn]['angular_velocity']) for fn in self.fixture_names}
        #TODO: Actually, this is only a temporary solution. We need to handle the case from the perception module
        # assert len(set(self.initial_fixture_velocities)) == 1, "All fixtures in a body must have the same initial velocity"
        self.initial_body_velocity = list(self.initial_fixture_velocities.values())[0]
        # assert len(set(self.initial_fixture_angular_velocities)) == 1, "All fixtures in a body must have the same initial angular velocity"
        self.initial_body_angular_velocity = list(self.initial_fixture_angular_velocities.values())[0]
        if len(set(self.initial_fixure_positions.values())) == 1:
            self.initial_body_position = list(self.initial_fixure_positions.values())[0]
        else:
            self.initial_body_position = b2Vec2(0., 0.)
            for fn in self.fixture_names:
                self.initial_body_position += self.initial_fixure_positions[fn]
            self.initial_body_position /= len(self.fixture_names)
        if len(set(self.initial_fixture_angles.values())) == 1:
            self.initial_body_angle = list(self.initial_fixture_angles.values())[0]
        else:
            self.initial_body_angle = 0.
        self.fixture_local_transforms = dict()
        for fn in self.fixture_names:
            self.fixture_local_transforms[fn] = {
                'position_diff': self.initial_fixure_positions[fn] - self.initial_body_position,
                'angle_diff': self.initial_fixture_angles[fn] - self.initial_body_angle,
            }

        body_def = b2BodyDef(
            position=self.initial_body_position,
            angle=float(self.initial_body_angle),
            type=BODYTYPE[self.body_type],
            # angularDamping=0.01,
        )
        body = world.CreateBody(body_def)
        for fn in self.fixture_names:
            fixture_definitions[fn].add_to_body(body, initial_state[fn], self.fixture_local_transforms[fn])
        body.linearVelocity = self.initial_body_velocity
        body.angularVelocity = self.initial_body_angular_velocity
        return body
    def update_state(self, state, body):
        for fn in self.fixture_names:
            state[fn]['position'] = tuple(body.transform * self.fixture_local_transforms[fn]['position_diff'])
            state[fn]['angle'] = body.angle + self.fixture_local_transforms[fn]['angle_diff']
            state[fn]['velocity'] = body.linearVelocity
            state[fn]['angular_velocity'] = body.angularVelocity

class JointDef:
    def __init__(self, name, body_names, joint_type, set_gt_anchor_for_debug=False,):
        self.name = name
        self.body_names = body_names
        assert len(self.body_names) == 2, "Joint must connect two bodies"
        self.joint_type = joint_type
        self.set_gt_anchor_for_debug = set_gt_anchor_for_debug
    def init(self, params, world, bodies):
        body1 = bodies[self.body_names[0]]
        body2 = bodies[self.body_names[1]]
        if self.joint_type == 'revolute':
            joint = world.CreateRevoluteJoint(
                bodyA=body1,
                bodyB=body2,
                anchor=(
                    body1.transform * b2Vec2(*(params.get_params(f'revjoint_anchor_{self.name}', 2)))
                    if not self.set_gt_anchor_for_debug
                    else body1.position
                ),
            )
        elif self.joint_type == 'distance':
            joint = world.CreateDistanceJoint(
                bodyA=body1,
                bodyB=body2,
                anchorA=body1.position,
                anchorB=body2.position,
            )
        elif self.joint_type == 'prismatic':
            joint = world.CreatePrismaticJoint(
                bodyA=body1,
                bodyB=body2,
                anchor=(
                    body1.transform * b2Vec2(*(params.get_params(f'prismjoint_anchor_{self.name}', 2)))
                    if not self.set_gt_anchor_for_debug
                    else body1.position
                ),
                axis=(
                    b2Vec2(*(params.get_params(f'prismatic_axis_{self.name}', 2)))
                    if not self.set_gt_anchor_for_debug
                    else b2Vec2(1, 0)
                ),
            )
        elif self.joint_type == 'weld':
            joint = world.CreateWeldJoint(
                bodyA=body1,
                bodyB=body2,
                localAnchorA=body1.transform * b2Vec2(*(params.get_params(f'weldjoint_anchor_{self.name}', 2))),
                localAnchorB=body2.transform * b2Vec2(*(params.get_params(f'weldjoint_anchor_{self.name}', 2))),
            )
        else:
            raise ValueError("Unknown joint type: {}".format(self.joint_type))
        return joint

class ContParams:
    def __init__(self, params):
        self.params = params
        self.name2indices = dict()
        self.index = 0
    def get_params(self, name, n=None):
        assert isinstance(name, str)
        if name in self.name2indices:
            if n is None:
                assert len(self.name2indices[name]) == 1, f"Name {name} is not unique"
                return self.params[self.name2indices[name][0]]
            else:
                assert len(self.name2indices[name]) == n, f"Name {name} is not unique"
                return [self.params[vi] for vi in self.name2indices[name]]
        if n is None:
            out = self.params[self.index]
            self.name2indices[name] = [self.index,]
            self.index += 1
        else:
            out = self.params[self.index:self.index+n]
            self.name2indices[name] = list(range(self.index, self.index+n))
            self.index += n
        return out
    def set_param(self, name, values):
        assert isinstance(name, str)
        assert name in self.name2indices, f"Name {name} not found"
        assert len(self.name2indices[name]) == len(values), f"Length mismatch: {len(self.name2indices[name])} != {len(values)}"
        for i, v in zip(self.name2indices[name], values):
            self.params[i] = v
    def set_params(self, params):
        for name, values in params.items():
            self.set_param(name, values)
    def get_indices(self, name):
        assert isinstance(name, str)
        assert name in self.name2indices, f"Name {name} not found"
        return tuple(self.name2indices[name])
    def __str__(self):
        return str({
            k: self.params[v[0]] if len(v) == 1 else [self.params[vi] for vi in v]
            for k, v in self.name2indices.items()
        })
    def __repr__(self):
        return str(self)
    def dumps(self,):
        return {
            name: (tuple(self.params[vi] for vi in v), tuple(v))
            for name, v in self.name2indices.items()
        }
    @staticmethod
    def loads(dump):
        param_num = max([i for _, (values, indices) in dump.items() for i in indices]) + 1
        out = ContParams(np.zeros(param_num))
        for name, (values, indices) in dump.items():
            out.name2indices[name] = list(indices)
            for i, v in zip(indices, values):
                out.params[i] = v
        return out
class Box2DProgram:
    def __init__(
        self,
        fixture_definitions,
        contactable_graph,
        body_definitions,
        joint_definitions,
        #TODO: Add action & reward related
    ):
        self.fixture_definitions = fixture_definitions
        self.contactable_graph = contactable_graph
        self.body_definitions = body_definitions
        self.joint_definitions = joint_definitions
    def simulate(self, params, initial_state, fps, max_time_steps, set_params=None):
        # ONLY FOR DEBUGGING
        assert params is not None, "Params must be provided"
        if set_params is not None:
            if isinstance(params, ContParams):
                params = params.params.copy()
            for index, value in set_params.items():
                for i, v in zip(index, value):
                    params[i] = v
        if not isinstance(params, ContParams):
            params = ContParams(params)
        return self._simulate(params, initial_state, fps, max_time_steps,)[0]
    def _simulate(self, params, initial_state, fps, max_time_steps,):
        gravity = params.get_params('gravity', 2)
        world = b2World(gravity=b2Vec2(*gravity), doSleep=True)

        fixture_definitions = {
            fn: f.init_def(params, categoryBits=self.contactable_graph.query_category(fn), maskBits=self.contactable_graph.query_mask(fn))
            for fn, f in self.fixture_definitions.items()
        }
        bodies = {
            bn:body.init(params, world, fixture_definitions, initial_state,)
            for bn, body in self.body_definitions.items()
        }
        joints = {
            jn: joint.init(params, world, bodies)
            for jn, joint in self.joint_definitions.items()
        }

        TARGET_FPS = fps
        TIME_STEP = 1.0 / TARGET_FPS
        out_states = [initial_state]
        for t in range(max_time_steps):
            #TODO: Add action & reward related
            world.Step(TIME_STEP, 10, 10)
            # STRIDE = TIME_STEP / (1./60)
            # assert abs(STRIDE-int(STRIDE)) < 1e-5, TARGET_FPS
            # for _ in range(int(STRIDE)):
                # world.Step(1./60, 15, 20)
            cur_state = copy.deepcopy(initial_state)
            for bn in bodies:
                self.body_definitions[bn].update_state(cur_state, bodies[bn],)
            out_states.append(cur_state)
            world.ClearForces()
        return out_states, params
    def _fit(self, trajs, fps, max_tries=100, maxiter=int(1e5), early_stop_threshold=0.1, loss_name='mse', optimizer='Nelder-Mead', verbose=False, set_params=None, batched=False,):
        if verbose:
            print('~'*50)
            print(f"Optimizing with {loss_name.upper()} loss: optimizer={optimizer}, early_stop_threshold={early_stop_threshold}, max_tries={max_tries}, maxiter={maxiter}")
        lowest_loss, best_res, best_params = 1e10, None, None
        for ti in range(max_tries):
            if verbose:
                print(f"Try {ti+1}/{max_tries}")
            params = np.random.rand(1000)
            res = minimize(loss_fn if not batched else batch_loss_fn, params, args=(self, trajs, fps, loss_name, set_params), method=optimizer, options={'maxiter': maxiter, 'disp': verbose})
            print(res)
            if res.fun < lowest_loss:
                lowest_loss = res.fun
                best_res = res
                best_params = res.x
            if res.fun < early_stop_threshold:
                assert lowest_loss < early_stop_threshold, (lowest_loss, early_stop_threshold)
                assert best_params is not None, "Best params is None"
                break;
        assert best_params is not None, "Best params is None"
        _, params = self._simulate(ContParams(best_params), trajs[0], fps, len(trajs)-1)
        if verbose:
            print(best_res)
            print(f"Best loss: {lowest_loss}")
            print(f"Best params: {params}")
        return params
    def fit(self, trajs, fps, hyperparams_list=({
        'max_tries': 100,
        'maxiter': int(1e5),
        'early_stop_threshold': 0.1,
        'loss_name': 'mse',
        'optimizer': 'Nelder-Mead',
    },), verbose=False, set_params=None, batched=False,):
        best_loss, best_params = 100, None
        for hyperparams in hyperparams_list:
            params = self._fit(trajs, fps, **hyperparams, verbose=verbose, set_params=set_params, batched=batched,)
            loss = loss_fn(params, self, trajs, fps, 'mse') # TODO: Set one unified loss function
            if loss < best_loss:
                best_loss = loss
                best_params = params
        return best_params

def _metrics_per_array_fn(true_array, pred_array):
    assert len(true_array) == len(pred_array), (len(true_array), len(pred_array))
    if np.isnan(true_array).any() or np.isnan(pred_array).any():
        return {
            'mse': 1000,
            'mae': 1000,
            'r2': -1000,
        }
    true_array = np.array(true_array)
    pred_array = np.array(pred_array)
    mse = np.mean((true_array - pred_array) ** 2)
    mae = np.mean(np.abs(true_array - pred_array))
    r2 = 1 - np.mean((true_array - pred_array) ** 2) / (
        np.mean((true_array - np.mean(true_array)) ** 2)
        if np.mean((true_array - np.mean(true_array)) ** 2) > 1e-2
        else 1e-2
    )
    assert r2 <= 1, (true_array, pred_array, r2)
    return {
        'mse': mse,
        'mae': mae,
        'r2': r2,
    }
def _metrics_fn(params, program, trajs, fps, set_params=None,):
    pred_trajs = program.simulate(params, trajs[0], fps, len(trajs)-1, set_params=set_params)
    assert len(pred_trajs) == len(trajs), (len(pred_trajs), len(trajs))

    true_array = {
        f'{fn}.position[{pi}]': [t[fn]['position'][pi] for t in trajs[1:]]
        for fn in trajs[0]
        for pi in range(2)
    }
    true_array.update({
        f'{fn}.angle': [t[fn]['angle'] for t in trajs[1:]]
        for fn in trajs[0]
    })

    pred_array = {
        f'{fn}.position[{pi}]': np.array([t[fn]['position'][pi] for t in pred_trajs[1:]])
        for fn in trajs[0]
        for pi in range(2)
    }
    pred_array.update({
        f'{fn}.angle': np.array([t[fn]['angle'] for t in pred_trajs[1:]])
        for fn in trajs[0]
    })

    metrics = {
        k: _metrics_per_array_fn(true_array[k], pred_array[k])
        for k in true_array
    }
    return metrics

LOSS_ITER = 0
def _loss_fn(params, program, trajs, fps, loss_name, set_params=None,):
    metrics = _metrics_fn(params, program, trajs, fps, set_params=set_params,)
    if loss_name == 'mse':
        loss = np.mean([v['mse'] for k, v in metrics.items()])
    elif loss_name == 'mae':
        loss = np.mean([v['mae'] for k, v in metrics.items()])
    elif loss_name == 'rmse':
        loss = np.mean([np.sqrt(v['mse']) for k, v in metrics.items()])
    elif loss_name == 'r2':
        loss = -np.mean([v['r2'] for k, v in metrics.items()])
    elif loss_name == 'areaweighted_mse':
        raise NotImplementedError
    else:
        raise ValueError(f"Unknown loss name: {loss_name}")
    global LOSS_ITER
    if LOSS_ITER % 1000 == 0:
        print(f"Iter {LOSS_ITER}: Loss={loss}")
    LOSS_ITER += 1
    return loss
def loss_fn(params, program, trajs, fps, loss_name, set_params=None,):
    try:
        return min(_loss_fn(params, program, trajs, fps, loss_name, set_params=set_params), 1000)
    except Exception as e:
        # raise e
        return 2000
BATCH_LOSS_ITER = 0
def batch_loss_fn(params, program, trajs, fps, loss_name, set_params=None,):
    assert isinstance(trajs, list) and all([isinstance(t, list) for t in trajs]), trajs
    losses = [loss_fn(params, program, t, fps, loss_name, set_params=set_params) for t in trajs]
    global BATCH_LOSS_ITER
    if BATCH_LOSS_ITER % 1000 == 0:
        print(f"Iter {BATCH_LOSS_ITER}: Losses={losses}")
    BATCH_LOSS_ITER += 1
    return np.mean(losses)
