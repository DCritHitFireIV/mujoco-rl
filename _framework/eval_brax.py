import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brax_common as bc
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.sac import networks as sac_networks
from train_brax import ppo_factory, sac_factory
p = argparse.ArgumentParser()
p.add_argument('--config', required=True)
p.add_argument('--params', default=None)
p.add_argument('--n-envs', type=int, default=None)
p.add_argument('--no-render', action='store_true')
p.add_argument('--output', default=None)
args = p.parse_args()
cfg = bc.load_config(args.config)
out = cfg['output_dir']
params = bc.load_params(args.params or os.path.join(out, 'best_params.pkl'))
raw_env = bc.make_raw_env(cfg)
if str(cfg.get('algo', 'ppo')).lower() == 'sac':
    nets = sac_factory(cfg)(raw_env.observation_size, raw_env.action_size)
    make_inference_fn = sac_networks.make_inference_fn(nets)
else:
    nets = ppo_factory(cfg)(raw_env.observation_size, raw_env.action_size)
    make_inference_fn = ppo_networks.make_inference_fn(nets)
result = bc.evaluate(cfg, make_inference_fn, params, n_envs=args.n_envs)
bc.write_json(result, os.path.join(out, 'eval_results.json'))
bc.write_episodes_csv(result, os.path.join(out, 'eval_metrics.csv'))
print(f"success={result['success']} ({result['success_reason']})")
print(f"mean_return={result['mean_return']:.2f} survival={result['survival']:.4f} mean_speed={result['mean_speed']} goal_dist={result['mean_goal_dist']} final_z={result['mean_final_z']}")
if not args.no_render:
    mp4 = args.output or os.path.join(out, 'best_rollout.mp4')
    bc.render_mp4(cfg, make_inference_fn, params, mp4, width=int(cfg.get('render', {}).get('width', 480)), height=int(cfg.get('render', {}).get('height', 480)))
    print(f'mp4: {mp4} ({os.path.getsize(mp4)} bytes)')
