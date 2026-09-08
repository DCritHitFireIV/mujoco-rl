from __future__ import annotations
import csv, json, os, platform
import numpy as np
import yaml
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
if platform.system() == 'Linux':
    os.environ.setdefault('MUJOCO_GL', 'osmesa')
import jax
import jax.numpy as jnp
from brax import envs
from brax.envs import _envs as BRAX_ENV_REGISTRY
from brax.io import model as io_model
if not hasattr(jax, 'device_put_replicated'):

    def _device_put_replicated(value, devices):
        return jax.tree_util.tree_map(lambda x: jax.device_put(jnp.expand_dims(x, 0), devices[0]), value)
    jax.device_put_replicated = _device_put_replicated

def patch_brax_unmap(module):

    def _unmap_fixed(v):

        def f(x):
            d = x.addressable_shards[0].data
            if d.ndim >= 1 and d.shape[0] == 1:
                return d[0]
            return d
        return jax.tree_util.tree_map(f, v)
    module._unpmap = _unmap_fixed

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_config(cfg, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True, default_flow_style=False)

def make_raw_env(cfg):
    if cfg.get('use_official_xml'):
        import brax_official_envs
        return brax_official_envs.make_official_env(cfg['env_name'], cfg.get('backend', 'mjx'))
    return BRAX_ENV_REGISTRY[cfg['env_name']](backend=cfg.get('backend', 'mjx'))

def save_params(path, params):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    io_model.save_params(path, params)
    return path

def load_params(path):
    return io_model.load_params(path)

def check_success(cfg, mean_return, survival, mean_speed, mean_goal_dist, mean_final_z=None):
    s = dict(cfg.get('success') or {})
    reasons = []
    if s.get('min_return') is not None and mean_return < float(s['min_return']):
        reasons.append(f"return {mean_return:.1f} < {s['min_return']}")
    if s.get('min_survival') is not None and survival < float(s['min_survival']):
        reasons.append(f"survival {survival:.3f} < {s['min_survival']}")
    if s.get('min_speed') is not None and (mean_speed is None or mean_speed < float(s['min_speed'])):
        reasons.append(f"speed {mean_speed} < {s['min_speed']}")
    if s.get('max_goal_dist') is not None and (mean_goal_dist is None or mean_goal_dist > float(s['max_goal_dist'])):
        reasons.append(f"goal_dist {mean_goal_dist:.4f} > {s['max_goal_dist']}")
    if s.get('min_final_z') is not None and (mean_final_z is None or mean_final_z < float(s['min_final_z'])):
        reasons.append(f"final_z {mean_final_z:.4f} < {s['min_final_z']}")
    ok = not reasons
    return (ok, 'ok' if ok else '; '.join(reasons))

def evaluate(cfg, make_inference_fn, params, n_envs=None, key_seed=1000):
    env = make_raw_env(cfg)
    max_steps = int(cfg.get('episode_length', 1000))
    n = int(n_envs or cfg.get('num_eval_envs', 128))
    policy = make_inference_fn(params, deterministic=True)
    want_z = cfg['env_name'] == 'humanoidstandup'

    def one_episode(rng):
        init = env.reset(rng)

        def body(carry, _):
            state, rew, length, speed_sum, dist_last, z_last, done_ever = carry
            act, _ = policy(state.obs, rng)
            nstate = env.step(state, act)
            alive = 1.0 - done_ever
            new_done_ever = jnp.maximum(done_ever, nstate.done)
            m = nstate.metrics
            speed = m.get('x_velocity', jnp.zeros_like(nstate.reward))
            dist = -m['reward_dist'] if 'reward_dist' in m else jnp.nan * jnp.ones_like(nstate.reward)
            dist = jnp.where(alive > 0, dist, dist_last)
            z = nstate.pipeline_state.q[2] if want_z else jnp.nan * jnp.ones_like(nstate.reward)
            z = jnp.where(alive > 0, z, z_last)
            return ((nstate, rew + alive * nstate.reward, length + alive, speed_sum + alive * speed, dist, z, new_done_ever), None)
        final, _ = jax.lax.scan(body, (init, jnp.array(0.0), jnp.array(0.0), jnp.array(0.0), jnp.nan * jnp.ones(()), jnp.nan * jnp.ones(()), jnp.array(0.0)), None, length=max_steps)
        _, rew, length, speed_sum, dist_last, z_last, done_ever = final
        terminated = jnp.where(length >= max_steps, 0.0, done_ever)
        survival = length / max_steps
        mean_speed = jnp.where(length > 0, speed_sum / jnp.maximum(length, 1.0), 0.0)
        return {'return': rew, 'length': length, 'terminated': terminated, 'survival': survival, 'mean_speed': mean_speed, 'goal_dist': dist_last, 'final_z': z_last}
    rngs = jax.random.split(jax.random.PRNGKey(key_seed), n)
    out = jax.jit(jax.vmap(one_episode))(rngs)
    out = {k: np.asarray(v) for k, v in out.items()}
    episodes = [{'episode': i, 'return': float(out['return'][i]), 'length': float(out['length'][i]), 'terminated': bool(out['terminated'][i] > 0.5), 'survival': float(out['survival'][i]), 'mean_speed': float(out['mean_speed'][i]), 'goal_dist': float(out['goal_dist'][i]) if np.isfinite(out['goal_dist'][i]) else None, 'final_z': float(out['final_z'][i]) if np.isfinite(out['final_z'][i]) else None} for i in range(n)]
    mean_return = float(np.mean(out['return']))
    survival = float(np.mean(out['survival']))
    sp = [e['mean_speed'] for e in episodes if e['mean_speed'] is not None]
    mean_speed = float(np.mean(sp)) if sp else None
    gd = [e['goal_dist'] for e in episodes if e['goal_dist'] is not None]
    mean_goal_dist = float(np.mean(gd)) if gd else None
    zs = [e['final_z'] for e in episodes if e['final_z'] is not None]
    mean_final_z = float(np.mean(zs)) if zs else None
    success, reason = check_success(cfg, mean_return, survival, mean_speed, mean_goal_dist, mean_final_z)
    score = -mean_goal_dist if (cfg.get('success') or {}).get('max_goal_dist') is not None else mean_return
    return {'env_name': cfg['env_name'], 'gymnasium_env_id': cfg.get('gymnasium_env_id'), 'engine': cfg.get('engine', 'brax-mjx'), 'n_episodes': n, 'mean_return': mean_return, 'survival': survival, 'mean_speed': mean_speed, 'mean_goal_dist': mean_goal_dist, 'mean_final_z': mean_final_z, 'success': success, 'success_reason': reason, 'score': score, 'episodes': episodes}

def rollout_trajectory(cfg, make_inference_fn, params, rng_key, max_steps=None):
    env = make_raw_env(cfg)
    max_steps = int(max_steps or cfg.get('episode_length', 1000))
    policy = make_inference_fn(params, deterministic=True)
    init = env.reset(rng_key)

    def body(state, _):
        act, _ = policy(state.obs, rng_key)
        return (env.step(state, act), None)

    def collect(state, _):
        nstate, _ = body(state, None)
        return (nstate, nstate)
    final, _ = jax.lax.scan(body, init, None, length=max_steps)
    _, traj = jax.lax.scan(collect, init, None, length=max_steps)
    return (init, traj, float(final.reward))

def render_mp4(cfg, make_inference_fn, params, output, rng_key=None, fps=None, width=480, height=480):
    import mujoco
    import imageio.v2 as imageio
    env = make_raw_env(cfg)
    rng = rng_key or jax.random.PRNGKey(int(cfg.get('seed', 0)) + 1234)
    init, traj, _ = rollout_trajectory(cfg, make_inference_fn, params, rng)
    model = env.sys.mj_model
    renderer = mujoco.Renderer(model, height=height, width=width)
    d = mujoco.MjData(model)
    render_cfg = dict(cfg.get('render') or {})
    camera = render_cfg.get('camera')
    if camera is None:
        try:
            names = [model.camera(i).name for i in range(model.ncam)]
            camera = 'track' if 'track' in names else -1
        except Exception:
            camera = -1

    def render_state(state):
        ps = state.pipeline_state
        d.qpos, d.qvel = (np.asarray(ps.q).reshape(-1), np.asarray(ps.qd).reshape(-1))
        mujoco.mj_forward(model, d)
        renderer.update_scene(d, camera=camera)
        return renderer.render()
    states = jax.tree.map(lambda x: np.asarray(x), (init, traj))
    frames = [render_state(states[0])]
    T = int(np.asarray(states[1].pipeline_state.q).shape[0])
    for t in range(T):
        s = jax.tree.map(lambda x: x[t], states[1])
        frames.append(render_state(s))
    fps = fps or int(render_cfg.get('fps', 30))
    out_dir = os.path.dirname(output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    imageio.mimsave(output, frames, fps=fps)
    return output

def write_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=float)

def write_episodes_csv(result, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['episode', 'return', 'length', 'terminated', 'survival', 'mean_speed', 'goal_dist', 'final_z'])
        for e in result['episodes']:
            w.writerow([e['episode'], f"{e['return']:.4f}", e['length'], e['terminated'], f"{e['survival']:.4f}", f"{e['mean_speed']:.4f}" if e['mean_speed'] is not None else '', f"{e['goal_dist']:.6f}" if e['goal_dist'] is not None else '', f"{e['final_z']:.4f}" if e['final_z'] is not None else ''])
