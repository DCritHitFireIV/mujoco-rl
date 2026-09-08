from __future__ import annotations
import argparse, csv, os, shutil, sys, time
from datetime import datetime
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def make_lr_schedule(ppo_cfg):
    lr = float(ppo_cfg['learning_rate'])
    if ppo_cfg.get('lr_schedule', 'linear') == 'linear':
        return lambda f: lr * (1.0 - f)
    return lr

def ppo_kwargs(cfg):
    pc = dict(cfg['ppo'])
    return dict(n_steps=int(pc.get('n_steps', 2048)), batch_size=int(pc.get('batch_size', 64)), n_epochs=int(pc.get('n_epochs', 10)), gamma=float(pc.get('gamma', 0.99)), gae_lambda=float(pc.get('gae_lambda', 0.95)), clip_range=float(pc.get('clip_range', 0.2)), ent_coef=float(pc.get('ent_coef', 0.0)), vf_coef=float(pc.get('vf_coef', 0.5)), max_grad_norm=float(pc.get('max_grad_norm', 0.5)), learning_rate=make_lr_schedule(pc), policy_kwargs=common.parse_policy_kwargs(pc.get('policy_kwargs')))

def build_model(cfg, env):
    from stable_baselines3 import PPO
    pc = dict(cfg['ppo'])
    tb_log = pc.get('tensorboard_log')
    if tb_log is None and cfg.get('train', {}).get('tensorboard', True):
        tb_log = os.path.join(cfg['output_dir'], 'tensorboard')
    return PPO(policy=pc.get('policy', 'MlpPolicy'), env=env, seed=int(cfg.get('seed', 0)), device=str(cfg.get('device', 'cuda')), verbose=0, tensorboard_log=tb_log, **ppo_kwargs(cfg))

def main():
    args = parse_args()
    cfg = common.load_config(args.config)
    out = cfg['output_dir']
    if args.dry_run:
        cfg.setdefault('train', {})['tensorboard'] = False
    else:
        os.makedirs(out, exist_ok=True)
        common.save_config(cfg, os.path.join(out, 'config.yaml'))
        log(f"env={cfg['env_id']} device={cfg.get('device')} budget={cfg['train']['total_timesteps']}")
    log('building vectorized env + VecNormalize ...')
    env = common.build_train_venv(cfg)
    latest_model = os.path.join(out, 'latest_model.zip')
    latest_vn = os.path.join(out, 'latest_vecnormalize.pkl')
    if args.dry_run:
        model = build_model(cfg, env)
        model.learn(total_timesteps=32, reset_num_timesteps=True, progress_bar=False)
        log('dry-run ok')
        env.close()
        return 0
    if args.resume and os.path.exists(latest_model):
        log(f'resume from {latest_model}')
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import VecNormalize
        base_env = common.build_base_venv(cfg)
        env = VecNormalize.load(latest_vn, base_env)
        env.training = True
        model = PPO.load(latest_model, env=env, device=cfg.get('device', 'cuda'), **ppo_kwargs(cfg))
    else:
        model = build_model(cfg, env)
    train_cfg = dict(cfg['train'])
    total = int(train_cfg['total_timesteps'])
    chunk = int(train_cfg.get('chunk', 2048))
    eval_every = int(train_cfg.get('eval_every', 128000))
    n_eval = int(train_cfg.get('n_eval_episodes', 10))
    patience = int(train_cfg.get('patience', 3))
    progress_csv = os.path.join(out, 'progress.csv')
    progress_exists = os.path.exists(progress_csv) and os.path.getsize(progress_csv) > 0
    with open(progress_csv, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if not progress_exists:
            w.writerow(['steps', 'elapsed_s', 'fps', 'mean_return', 'survival', 'mean_speed', 'mean_goal_dist', 'mean_final_z_dist', 'success', 'score', 'best_score', 'note'])
    best_score = -1e+18
    best_model = os.path.join(out, 'best_model.zip')
    best_vn = os.path.join(out, 'best_vecnormalize.pkl')
    steps_done = int(model.num_timesteps)
    last_eval_steps = steps_done
    start_wall = time.time()
    if progress_exists and steps_done > 0:
        try:
            with open(progress_csv, encoding='utf-8') as f:
                rows = list(csv.reader(f))
            if len(rows) > 1 and rows[-1][1]:
                start_wall = time.time() - float(rows[-1][1])
        except Exception:
            pass
    evals_since_best = 0
    success_ever = False
    while steps_done < total:
        prev_steps = int(model.num_timesteps)
        t0 = time.time()
        model.learn(total_timesteps=chunk, reset_num_timesteps=False, tb_log_name=cfg['env_id'], progress_bar=False)
        dt = max(time.time() - t0, 1e-06)
        steps_done = int(model.num_timesteps)
        delta = max(steps_done - prev_steps, 1)
        fps = delta / dt
        elapsed = time.time() - start_wall
        model.save(latest_model)
        env.save(latest_vn)
        with open(progress_csv, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([steps_done, f'{elapsed:.1f}', f'{fps:.1f}', '', '', '', '', '', '', '', f'{best_score:.4f}', ''])
        log(f'steps={steps_done}/{total} elapsed={elapsed:.0f}s fps={fps:.0f}')
        do_eval = steps_done - last_eval_steps >= eval_every or steps_done >= total
        if not do_eval:
            continue
        last_eval_steps = steps_done
        log(f'evaluating ({n_eval} eps deterministic) ...')
        result = common.evaluate(cfg, model, latest_vn, n_episodes=n_eval)
        log(f"eval return={result['mean_return']:.1f} survival={result['survival']:.3f} speed={result['mean_speed']} goal_dist={result['mean_goal_dist']} success={result['success']} ({result['success_reason']})")
        with open(progress_csv, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([steps_done, f'{elapsed:.1f}', f'{fps:.1f}', f"{result['mean_return']:.2f}", f"{result['survival']:.4f}", f"{result['mean_speed']:.4f}" if result['mean_speed'] is not None else '', f"{result['mean_goal_dist']:.6f}" if result['mean_goal_dist'] is not None else '', f"{result['mean_final_z_dist']:.4f}" if result['mean_final_z_dist'] is not None else '', result['success'], f"{result['score']:.2f}", f'{best_score:.4f}', 'eval'])
        if result['success']:
            success_ever = True
            if result['score'] > best_score:
                best_score = result['score']
                shutil.copyfile(latest_model, best_model)
                shutil.copyfile(latest_vn, best_vn)
                evals_since_best = 0
                log(f'NEW BEST score={best_score:.2f}')
            else:
                evals_since_best += 1
        else:
            evals_since_best += 1
        if success_ever and evals_since_best >= patience:
            log('success + patience -> early stop')
            break
    env.close()
    from stable_baselines3 import PPO
    final_model_path = best_model if os.path.exists(best_model) else latest_model
    final_vn_path = best_vn if os.path.exists(best_vn) else latest_vn
    final_model = PPO.load(final_model_path, device=cfg.get('device', 'auto'))
    log(f'final evaluation using {os.path.basename(final_model_path)} ...')
    final_eval = common.evaluate(cfg, final_model, final_vn_path, n_episodes=n_eval)
    common.write_json(final_eval, os.path.join(out, 'eval_results.json'))
    common.write_episodes_csv(final_eval, os.path.join(out, 'eval_metrics.csv'))
    log(f"eval_results.json: success={final_eval['success']} return={final_eval['mean_return']:.1f} survival={final_eval['survival']:.3f} speed={final_eval['mean_speed']}")
    mp4_path = os.path.join(out, 'best_rollout.mp4')
    try:
        common.render_rollout(cfg, final_model, final_vn_path, mp4_path)
        log(f'mp4 rendered: {mp4_path} ({os.path.getsize(mp4_path)} bytes)')
    except Exception as e:
        log(f'WARNING mp4 render failed: {type(e).__name__}: {e}')
    if final_eval['success']:
        log(f"TRAINING SUCCESS for {cfg['env_id']}")
        return 0
    log(f"TRAINING FINISHED (not reached) for {cfg['env_id']}: {final_eval['success_reason']}")
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
