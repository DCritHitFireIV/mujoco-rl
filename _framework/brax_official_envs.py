from __future__ import annotations
import os
import jax
import jax.numpy as jnp
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
_ASSET_DIR = '/opt/train-venv/lib/python3.11/site-packages/gymnasium/envs/mujoco/assets'

def _load(xml_name):
    return mjcf.load(os.path.join(_ASSET_DIR, xml_name))

class OfficialHopper(PipelineEnv):

    def __init__(self, forward_reward_weight=1.0, ctrl_cost_weight=0.001, healthy_reward=1.0, terminate_when_unhealthy=True, healthy_state_range=(-100.0, 100.0), healthy_z_range=(0.7, float('inf')), healthy_angle_range=(-0.2, 0.2), reset_noise_scale=0.005, exclude_current_positions_from_observation=True, backend='mjx', **kwargs):
        sys = _load('hopper.xml')
        n_frames = 4
        kwargs['n_frames'] = kwargs.get('n_frames', n_frames)
        super().__init__(sys=sys, backend=backend, **kwargs)
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_state_range = healthy_state_range
        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation

    @property
    def observation_size(self):
        return 11

    @property
    def action_size(self):
        return 3

    @property
    def backend(self):
        return self._backend

    def _is_healthy(self, ps):
        z, angle = (ps.q[1], ps.q[2])
        state = jnp.concatenate([ps.q[2:], ps.qd])
        mn, mx = self._healthy_state_range
        mz, Mz = self._healthy_z_range
        ma, Ma = self._healthy_angle_range
        healthy_state = jnp.all(jnp.logical_and(mn < state, state < mx))
        healthy_z = jnp.logical_and(mz < z, z < Mz)
        healthy_angle = jnp.logical_and(ma < angle, angle < Ma)
        return jnp.logical_and(jnp.logical_and(healthy_state, healthy_z), healthy_angle)

    def reset(self, rng):
        qpos = self.sys.init_q + jax.random.uniform(rng, (self.sys.q_size(),), minval=-self._reset_noise_scale, maxval=self._reset_noise_scale)
        qvel = jax.random.uniform(rng, (self.sys.qd_size(),), minval=-self._reset_noise_scale, maxval=self._reset_noise_scale)
        ps = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(ps)
        reward, done, zero = jnp.zeros(3)
        metrics = {'reward_forward': zero, 'reward_ctrl': zero, 'reward_survive': zero, 'x_position': zero, 'z_distance_from_origin': zero, 'x_velocity': zero}
        return State(ps, obs, reward, done, metrics)

    def step(self, state, action):
        ps0 = state.pipeline_state
        ps = self.pipeline_step(ps0, action)
        x_velocity = (ps.q[0] - ps0.q[0]) / self.dt
        obs = self._get_obs(ps)
        is_healthy = self._is_healthy(ps)
        healthy_reward = is_healthy * self._healthy_reward
        forward_reward = self._forward_reward_weight * x_velocity
        ctrl_cost = self._ctrl_cost_weight * jnp.sum(jnp.square(action))
        reward = forward_reward + healthy_reward - ctrl_cost
        terminated = jnp.where(jnp.logical_not(is_healthy) & self._terminate_when_unhealthy, 1.0, 0.0)
        state.metrics.update(reward_forward=forward_reward, reward_ctrl=-ctrl_cost, reward_survive=healthy_reward, x_position=ps.q[0], z_distance_from_origin=ps.q[1] - self.sys.init_q[1], x_velocity=x_velocity)
        return state.replace(pipeline_state=ps, obs=obs, reward=reward, done=terminated)

    def _get_obs(self, ps):
        position = ps.q
        velocity = jnp.clip(ps.qd, -10, 10)
        if self._exclude_current_positions_from_observation:
            position = position[1:]
        return jnp.concatenate((position, velocity))

class OfficialHalfCheetah(PipelineEnv):

    def __init__(self, forward_reward_weight=1.0, ctrl_cost_weight=0.1, reset_noise_scale=0.1, exclude_current_positions_from_observation=True, backend='mjx', **kwargs):
        sys = _load('half_cheetah.xml')
        n_frames = 5
        kwargs['n_frames'] = kwargs.get('n_frames', n_frames)
        super().__init__(sys=sys, backend=backend, **kwargs)
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation

    @property
    def observation_size(self):
        return 17

    @property
    def action_size(self):
        return 6

    @property
    def backend(self):
        return self._backend

    def reset(self, rng):
        qpos = self.sys.init_q + jax.random.uniform(rng, (self.sys.q_size(),), minval=-self._reset_noise_scale, maxval=self._reset_noise_scale)
        qvel = jax.random.uniform(rng, (self.sys.qd_size(),), minval=-self._reset_noise_scale, maxval=self._reset_noise_scale)
        ps = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(ps)
        reward, done, zero = jnp.zeros(3)
        metrics = {'reward_forward': zero, 'reward_ctrl': zero, 'x_position': zero, 'x_velocity': zero}
        return State(ps, obs, reward, done, metrics)

    def step(self, state, action):
        ps0 = state.pipeline_state
        ps = self.pipeline_step(ps0, action)
        x_velocity = (ps.q[0] - ps0.q[0]) / self.dt
        forward_reward = self._forward_reward_weight * x_velocity
        ctrl_cost = self._ctrl_cost_weight * jnp.sum(jnp.square(action))
        obs = self._get_obs(ps)
        reward = forward_reward - ctrl_cost
        state.metrics.update(reward_forward=forward_reward, reward_ctrl=-ctrl_cost, x_position=ps.q[0], x_velocity=x_velocity)
        return state.replace(pipeline_state=ps, obs=obs, reward=reward, done=jnp.zeros_like(reward))

    def _get_obs(self, ps):
        position = ps.q
        velocity = ps.qd
        if self._exclude_current_positions_from_observation:
            position = position[1:]
        return jnp.concatenate((position, velocity))

class OfficialSwimmer(PipelineEnv):

    def __init__(self, forward_reward_weight=1.0, ctrl_cost_weight=0.0001, reset_noise_scale=0.1, exclude_current_positions_from_observation=True, backend='mjx', **kwargs):
        sys = _load('swimmer.xml')
        n_frames = 4
        kwargs['n_frames'] = kwargs.get('n_frames', n_frames)
        super().__init__(sys=sys, backend=backend, **kwargs)
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation

    @property
    def observation_size(self):
        return 8

    @property
    def action_size(self):
        return 2

    @property
    def backend(self):
        return self._backend

    def reset(self, rng):
        qpos = self.sys.init_q + jax.random.uniform(rng, (self.sys.q_size(),), minval=-self._reset_noise_scale, maxval=self._reset_noise_scale)
        qvel = jax.random.uniform(rng, (self.sys.qd_size(),), minval=-self._reset_noise_scale, maxval=self._reset_noise_scale)
        ps = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(ps)
        reward, done, zero = jnp.zeros(3)
        metrics = {'reward_forward': zero, 'reward_ctrl': zero, 'x_position': zero, 'y_position': zero, 'distance_from_origin': zero, 'x_velocity': zero, 'y_velocity': zero}
        return State(ps, obs, reward, done, metrics)

    def step(self, state, action):
        ps0 = state.pipeline_state
        ps = self.pipeline_step(ps0, action)
        before = ps0.q[:2]
        after = ps.q[:2]
        x_velocity = (after[0] - before[0]) / self.dt
        y_velocity = (after[1] - before[1]) / self.dt
        forward_reward = self._forward_reward_weight * x_velocity
        ctrl_cost = self._ctrl_cost_weight * jnp.sum(jnp.square(action))
        obs = self._get_obs(ps)
        reward = forward_reward - ctrl_cost
        state.metrics.update(reward_forward=forward_reward, reward_ctrl=-ctrl_cost, x_position=after[0], y_position=after[1], distance_from_origin=jnp.linalg.norm(after), x_velocity=x_velocity, y_velocity=y_velocity)
        return state.replace(pipeline_state=ps, obs=obs, reward=reward, done=jnp.zeros_like(reward))

    def _get_obs(self, ps):
        position = ps.q
        velocity = ps.qd
        if self._exclude_current_positions_from_observation:
            position = position[2:]
        return jnp.concatenate((position, velocity))
OFFICIAL_ENVS = {'hopper': OfficialHopper, 'halfcheetah': OfficialHalfCheetah, 'swimmer': OfficialSwimmer}

def make_official_env(env_name, backend='mjx'):
    if env_name not in OFFICIAL_ENVS:
        raise KeyError(f'official xml env not implemented: {env_name}')
    return OFFICIAL_ENVS[env_name](backend=backend)
