from __future__ import annotations
import argparse, csv, os, shutil, sys, time
from datetime import datetime
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brax_common as bc
from brax.training.agents.ppo import networks, train as ppo_train
from brax.training.agents.sac import networks as sac_networks, train as sac_train
bc.patch_brax_unmap(ppo_train)
bc.patch_brax_unmap(sac_train)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def _norm(cfg):
    if bool(cfg.get('normalize_observations', True)):
        import importlib
        rs = importlib.import_module('brax.training.acme.running_statistics')
        return rs.normalize
    return None

def ppo_factory(cfg):
    pc = dict(cfg.get('ppo') or {})
    ps = tuple((int(x) for x in pc.get('policy_hidden_layer_sizes', [64, 64, 64, 64])))
    vs = tuple((int(x) for x in pc.get('value_hidden_layer_sizes', [256, 256, 256, 256, 256])))
    default_norm = _norm(cfg)

    def factory(obs, act, preprocess_observations_fn=None):
        if preprocess_observations_fn is None:
            preprocess_observations_fn = default_norm
        return networks.make_ppo_networks(obs, act, preprocess_observations_fn=preprocess_observations_fn, policy_hidden_layer_sizes=ps, value_hidden_layer_sizes=vs)
    return factory

def sac_factory(cfg):
    sc = dict(cfg.get('sac') or {})
    sizes = tuple((int(x) for x in sc.get('hidden_layer_sizes', [256, 256])))
    default_norm = _norm(cfg)

    def factory(obs, act, preprocess_observations_fn=None):
        if preprocess_observations_fn is None:
            preprocess_observations_fn = default_norm
        return sac_networks.make_sac_networks(obs, act, preprocess_observations_fn=preprocess_observations_fn, hidden_layer_sizes=sizes)
    return factory

def main():
    args = parse_args()
    cfg = bc.load_config(args.config)
    out = cfg['output_dir']
    if not args.dry_run:
        os.makedirs(out, exist_ok=True)
        bc.save_config(cfg, os.path.join(out, 'config.yaml'))
        log(f"env={cfg['env_name']} backend={cfg.get('backend')} algo={cfg.get('algo', 'ppo')} num_envs={cfg.get('num_envs')} budget={cfg.get('total_timesteps')}")
    log('building raw brax env (GPU physics) ...')
    raw_env = bc.make_raw_env(cfg)
    if args.dry_run:
        import jax
        key = jax.random.PRNGKey(0)
        state = jax.jit(raw_env.reset)(key)
        act = jax.random.uniform(key, (raw_env.action_size,), minval=-1, maxval=1)
        state = jax.jit(raw_env.step)(state, act)
        log(f'dry-run ok: obs={raw_env.observation_size} act={raw_env.action_size} reward={float(state.reward):.3f}')
        return 0
    algo = str(cfg.get('algo', 'ppo')).lower()
    num_evals = int(cfg.get('num_evals', 21))
    ckpt_dir = os.path.join(out, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    restore_params = None
    if args.resume:
        latest = os.path.join(out, 'latest_params.pkl')
        if os.path.exists(latest):
            restore_params = bc.load_params(latest)
            log(f'resume weights from {latest}')
    progress_csv = os.path.join(out, 'progress.csv')
    progress_exists = os.path.exists(progress_csv) and os.path.getsize(progress_csv) > 0
    pf = open(progress_csv, 'a', newline='', encoding='utf-8')
    w = csv.writer(pf)
    if not progress_exists:
        w.writerow(['steps', 'wall_s', 'fps', 'eval_return', 'eval_return_std', 'best_return', 'note'])
    best_return = -1e+18
    best_step = None
    start_wall = time.time()

    def progress_fn(step, metrics):
        nonlocal best_return, best_step
        m = {}
        for k, v in metrics.items():
            try:
                a = np.asarray(v)
                if a.size == 1:
                    m[k] = float(a)
            except Exception:
                pass
        ret = m.get('eval/episode_reward')
        std = m.get('eval/episode_reward_std')
        sps = m.get('training/sps')
        wall = time.time() - start_wall
        w.writerow([int(step), f'{wall:.1f}', f'{sps:.0f}' if sps else '', f'{ret:.2f}' if ret is not None else '', f'{std:.3f}' if std is not None else '', f'{best_return:.2f}', 'eval'])
        pf.flush()
        log(f'steps={step} fps={sps or 0:.0f} eval_return={ret} std={std} best={best_return:.2f}')
        if ret is not None and ret > best_return:
            best_return = ret
            best_step = int(step)

    def ppo_policy_params_fn(step, make_policy, params):
        step = int(step)
        path = os.path.join(ckpt_dir, f'params_{step}.pkl')
        bc.save_params(path, params)
        shutil.copyfile(path, os.path.join(out, 'latest_params.pkl'))
        log(f'saved params step={step}')
    log(f'starting brax {algo.upper()} training ...')
    t0 = time.time()
    sac_logdir = None
    pc = dict(cfg.get('ppo') or {})
    sc = dict(cfg.get('sac') or {})
    if algo == 'sac':
        sac_logdir = os.path.join(ckpt_dir, 'sac')
        make_inference_fn, final_params, _ = sac_train.train(environment=raw_env, num_timesteps=int(cfg['total_timesteps']), episode_length=int(cfg.get('episode_length', 1000)), wrap_env=True, action_repeat=int(cfg.get('action_repeat', 1)), num_envs=int(cfg.get('num_envs', 2048)), num_eval_envs=int(cfg.get('num_eval_envs', 128)), learning_rate=float(sc.get('learning_rate', 0.0003)), discounting=float(sc.get('discounting', 0.97)), reward_scaling=float(sc.get('reward_scaling', 10.0)), tau=float(sc.get('tau', 0.005)), batch_size=int(sc.get('batch_size', 256)), grad_updates_per_step=int(sc.get('grad_updates_per_step', 1)), min_replay_size=int(sc.get('min_replay_size', 0)), max_replay_size=int(sc['max_replay_size']) if sc.get('max_replay_size') is not None else None, normalize_observations=bool(cfg.get('normalize_observations', True)), deterministic_eval=bool(cfg.get('deterministic_eval', True)), num_evals=num_evals, progress_fn=progress_fn, checkpoint_logdir=sac_logdir, network_factory=sac_factory(cfg), seed=int(cfg.get('seed', 0)))
    else:
        make_inference_fn, final_params, _ = ppo_train.train(environment=raw_env, num_timesteps=int(cfg['total_timesteps']), num_envs=int(cfg.get('num_envs', 2048)), episode_length=int(cfg.get('episode_length', 1000)), action_repeat=int(cfg.get('action_repeat', 1)), wrap_env=True, learning_rate=float(pc.get('learning_rate', 0.0003)), entropy_cost=float(pc.get('entropy_cost', 0.001)), discounting=float(pc.get('discounting', 0.97)), unroll_length=int(pc.get('unroll_length', 10)), batch_size=int(pc.get('batch_size', 512)), num_minibatches=int(pc.get('num_minibatches', 40)), num_updates_per_batch=int(pc.get('num_updates_per_batch', 4)), normalize_observations=bool(cfg.get('normalize_observations', True)), reward_scaling=float(pc.get('reward_scaling', 1.0)), clipping_epsilon=float(pc.get('clipping_epsilon', 0.3)), gae_lambda=float(pc.get('gae_lambda', 0.95)), max_grad_norm=float(pc['max_grad_norm']) if pc.get('max_grad_norm') is not None else None, normalize_advantage=bool(pc.get('normalize_advantage', True)), vf_loss_coefficient=float(pc.get('vf_loss_coefficient', 0.5)), num_evals=num_evals, num_eval_envs=int(cfg.get('num_eval_envs', 128)), deterministic_eval=bool(cfg.get('deterministic_eval', True)), progress_fn=progress_fn, policy_params_fn=ppo_policy_params_fn, save_checkpoint_path=os.path.join(ckpt_dir, 'brax_ckpt'), restore_params=restore_params, network_factory=ppo_factory(cfg), seed=int(cfg.get('seed', 0)), run_evals=True)
    train_wall = time.time() - t0
    pf.close()
    log(f'training loop finished in {train_wall:.0f}s; best_step={best_step}')
    best_pkl = os.path.join(out, 'best_params.pkl')
    if algo == 'sac' and sac_logdir and (best_step is not None):
        from brax.training.agents.sac import checkpoint as sac_checkpoint
        ckpt_path = os.path.join(sac_logdir, f'{best_step:012d}')
        if os.path.isdir(ckpt_path):
            best_params = sac_checkpoint.load(ckpt_path)
            bc.save_params(best_pkl, best_params)
            log(f'best SAC params step={best_step}')
        else:
            best_params = final_params
            bc.save_params(best_pkl, final_params)
    else:
        chosen = os.path.join(ckpt_dir, f'params_{best_step}.pkl') if best_step is not None else None
        if chosen and os.path.exists(chosen):
            shutil.copyfile(chosen, best_pkl)
            log(f'best params step={best_step} -> best_params.pkl')
        else:
            bc.save_params(best_pkl, final_params)
            log('using final params as best')
    bc.save_params(os.path.join(out, 'latest_params.pkl'), final_params)
    best_params = bc.load_params(best_pkl)
    log('running custom deterministic eval (128 envs) ...')
    eval_result = bc.evaluate(cfg, make_inference_fn, best_params, n_envs=int(cfg.get('num_eval_envs', 128)))
    bc.write_json(eval_result, os.path.join(out, 'eval_results.json'))
    bc.write_episodes_csv(eval_result, os.path.join(out, 'eval_metrics.csv'))
    log(f"eval success={eval_result['success']} ({eval_result['success_reason']}) return={eval_result['mean_return']:.2f} survival={eval_result['survival']:.3f} speed={eval_result['mean_speed']} goal_dist={eval_result['mean_goal_dist']} final_z={eval_result['mean_final_z']}")
    mp4_path = os.path.join(out, 'best_rollout.mp4')
    try:
        bc.render_mp4(cfg, make_inference_fn, best_params, mp4_path, width=int(cfg.get('render', {}).get('width', 480)), height=int(cfg.get('render', {}).get('height', 480)))
        log(f'mp4 rendered: {mp4_path} ({os.path.getsize(mp4_path)} bytes)')
    except Exception as e:
        log(f'WARNING mp4 render failed: {type(e).__name__}: {e}')
    if eval_result['success']:
        log(f"TRAINING SUCCESS for {cfg['env_name']}")
    else:
        log(f"TRAINING FINISHED (not reached) for {cfg['env_name']}: {eval_result['success_reason']}")
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
