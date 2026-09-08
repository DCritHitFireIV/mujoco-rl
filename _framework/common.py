from __future__ import annotations
import csv, json, os, platform
import numpy as np
import yaml

def set_render_backend():
    if platform.system() == 'Linux':
        os.environ.setdefault('MUJOCO_GL', 'osmesa')

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_config(cfg, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True, default_flow_style=False)

def parse_policy_kwargs(pk):
    import torch.nn as nn
    out = dict(pk or {})
    if 'activation_fn' in out and isinstance(out['activation_fn'], str):
        out['activation_fn'] = {'tanh': nn.Tanh, 'relu': nn.ReLU}[out['activation_fn'].lower()]
    if 'net_arch' in out:
        out['net_arch'] = [int(x) for x in out['net_arch']]
    return out

def make_env_thunk(env_id, seed, env_kwargs=None):

    def _f():
        import gymnasium as gym
        env = gym.make(env_id, **env_kwargs or {})
        env.reset(seed=seed)
        return env
    return _f

def build_base_venv(cfg):
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    env_id = cfg['env_id']
    n = int(cfg.get('n_envs', 12))
    seed = int(cfg.get('seed', 0))
    env_kwargs = dict(cfg.get('env_kwargs') or {})
    thunks = [make_env_thunk(env_id, seed + i, env_kwargs) for i in range(n)]
    if n > 1:
        return SubprocVecEnv(thunks, start_method=str(cfg.get('start_method', 'fork')))
    return DummyVecEnv(thunks)

def build_train_venv(cfg):
    from stable_baselines3.common.vec_env import VecNormalize
    vec = build_base_venv(cfg)
    vc = dict(cfg.get('vecnormalize') or {})
    return VecNormalize(vec, training=True, norm_obs=bool(vc.get('norm_obs', True)), norm_reward=bool(vc.get('norm_reward', True)), gamma=float(vc.get('gamma', 0.99)), clip_obs=float(vc.get('clip_obs', 10.0)))

def make_single_env(cfg, seed, render_mode=None):
    import gymnasium as gym
    env = gym.make(cfg['env_id'], render_mode=render_mode, **cfg.get('env_kwargs') or {})
    env.reset(seed=seed)
    return env

def wrap_single_env(env, cfg, vecnorm_path=None):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    vc = dict(cfg.get('vecnormalize') or {})
    venv = VecNormalize(DummyVecEnv([lambda: env]), training=False, norm_obs=bool(vc.get('norm_obs', True)), norm_reward=False, gamma=float(vc.get('gamma', 0.99)), clip_obs=float(vc.get('clip_obs', 10.0)))
    if vecnorm_path and os.path.exists(vecnorm_path):
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False
    return venv

def _goal_dist(cfg, env):
    env_id = cfg['env_id']
    try:
        u = env.unwrapped
        if env_id.startswith('Pusher'):
            obj = np.asarray(u.get_body_com('object'))[:2]
            goal = np.asarray(u.get_body_com('goal'))[:2]
            return float(np.linalg.norm(obj - goal))
        if env_id.startswith('Reacher'):
            tip = np.asarray(u.get_body_com('fingertip'))[:2]
            target = np.asarray(u.get_body_com('target'))[:2]
            return float(np.linalg.norm(tip - target))
    except Exception:
        pass
    return None

def evaluate(cfg, model, vecnorm_path=None, n_episodes=None, render_episode=None, render_output=None):
    set_render_backend()
    import imageio.v2 as imageio
    train_cfg = cfg.get('train', {})
    n_episodes = int(n_episodes or train_cfg.get('n_eval_episodes', 10))
    render_cfg = dict(cfg.get('render') or {})
    episodes = []
    frames = []
    written_mp4 = None
    for ep in range(n_episodes):
        seed = int(cfg.get('eval_seed', 1000)) + ep
        render_mode = 'rgb_array' if render_episode is not None and ep == render_episode else None
        env = make_single_env(cfg, seed, render_mode)
        max_steps = int(cfg.get('max_episode_steps') or env.spec.max_episode_steps)
        venv = wrap_single_env(env, cfg, vecnorm_path)
        obs = venv.reset()
        ep_return = 0.0
        ep_len = 0
        speeds = []
        goal_dists = []
        z_dists = []
        terminated = False
        truncation = False
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = venv.step(action)
            info = infos[0] if infos else {}
            ep_return += float(np.asarray(rewards).reshape(-1)[0])
            ep_len += 1
            if 'x_velocity' in info:
                speeds.append(float(info['x_velocity']))
            if 'z_distance_from_origin' in info:
                z_dists.append(float(info['z_distance_from_origin']))
            gd = _goal_dist(cfg, env)
            if gd is not None:
                goal_dists.append(gd)
            if render_mode is not None:
                frames.append(np.asarray(env.render()))
            if bool(np.asarray(dones).reshape(-1)[0]):
                if info.get('TimeLimit.truncated', False):
                    truncation = True
                else:
                    terminated = True
                break
        episodes.append({'episode': ep, 'seed': seed, 'return': ep_return, 'length': ep_len, 'terminated': terminated, 'truncated': truncation, 'survival': ep_len / max_steps if max_steps else 0.0, 'mean_speed': float(np.mean(speeds)) if speeds else None, 'goal_dist': goal_dists[-1] if goal_dists else None, 'final_z_dist': z_dists[-1] if z_dists else None})
        if render_mode is not None and frames:
            fps = float(render_cfg.get('fps') or env.metadata.get('render_fps', 30))
            imageio.mimsave(render_output, frames, fps=fps)
            written_mp4 = render_output
        env.close()
    mean_return = float(np.mean([e['return'] for e in episodes]))
    survival = float(np.mean([e['survival'] for e in episodes]))
    speeds = [e['mean_speed'] for e in episodes if e['mean_speed'] is not None]
    mean_speed = float(np.mean(speeds)) if speeds else None
    gds = [e['goal_dist'] for e in episodes if e['goal_dist'] is not None]
    mean_goal_dist = float(np.mean(gds)) if gds else None
    zs = [e['final_z_dist'] for e in episodes if e['final_z_dist'] is not None]
    mean_final_z_dist = float(np.mean(zs)) if zs else None
    success, reason = check_success(cfg, mean_return, survival, mean_speed, mean_goal_dist, mean_final_z_dist)
    score = -mean_goal_dist if (cfg.get('success') or {}).get('max_goal_dist') is not None else mean_return
    return {'env_id': cfg['env_id'], 'engine': cfg.get('engine', 'gymnasium-sb3'), 'n_episodes': n_episodes, 'mean_return': mean_return, 'survival': survival, 'mean_speed': mean_speed, 'mean_goal_dist': mean_goal_dist, 'mean_final_z_dist': mean_final_z_dist, 'success': success, 'success_reason': reason, 'score': score, 'episodes': episodes, 'render_mp4': written_mp4}

def check_success(cfg, mean_return, survival, mean_speed, mean_goal_dist, mean_final_z_dist=None):
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
    if s.get('min_final_z_dist') is not None and (mean_final_z_dist is None or mean_final_z_dist < float(s['min_final_z_dist'])):
        reasons.append(f"final_z_dist {mean_final_z_dist:.4f} < {s['min_final_z_dist']}")
    ok = not reasons
    return (ok, 'ok' if ok else '; '.join(reasons))

def render_rollout(cfg, model, vecnorm_path, output, episode_seed=None):
    result = evaluate(cfg, model, vecnorm_path, n_episodes=1, render_episode=0, render_output=output)
    if not result['render_mp4'] or not os.path.exists(result['render_mp4']):
        raise RuntimeError('mp4 渲染失败')
    return result['render_mp4']

def write_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=float)

def write_episodes_csv(result, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['episode', 'seed', 'return', 'length', 'terminated', 'truncated', 'survival', 'mean_speed', 'goal_dist', 'final_z_dist'])
        for e in result['episodes']:
            w.writerow([e['episode'], e['seed'], f"{e['return']:.4f}", e['length'], e['terminated'], e['truncated'], f"{e['survival']:.4f}", f"{e['mean_speed']:.4f}" if e['mean_speed'] is not None else '', f"{e['goal_dist']:.6f}" if e['goal_dist'] is not None else '', f"{e['final_z_dist']:.4f}" if e['final_z_dist'] is not None else ''])
