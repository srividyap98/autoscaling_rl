"""Component ablations over paired training seeds.

Run python -m autoscaling.ablation --steps 50000 --seeds 10 --jobs 4

Variants share seeds, so seed s starts every variant from the same network
weights and training traffic; differences are reported as paired per-seed
differences from the full method.
  full       clipped PPO with GAE(lambda=0.95)
  no_clip    unclipped surrogate (clip=inf), same epochs and minibatches
  no_gae_mc  lambda=1: Monte Carlo return minus value baseline
  no_gae_td  lambda=0: one-step TD advantage
"""
import argparse, csv, json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .experiment import OUT, TEST, evaluate, train, writecsv
from .ppo import PPO
from .stats import ci95
ABLATION = OUT / 'ablations'
VARIANTS = {'full': {}, 'no_clip': dict(clip=np.inf), 'no_gae_mc': dict(lam=1.0), 'no_gae_td': dict(lam=0.0)}
LABELS = {'full': 'Full PPO (clip 0.2, GAE 0.95)', 'no_clip': 'No clipping', 'no_gae_mc': 'No GAE: Monte Carlo (λ=1)', 'no_gae_td': 'No GAE: one-step TD (λ=0)'}
COLORS = {'full': '#dc7841', 'no_clip': '#3b6fb6', 'no_gae_mc': '#21867a', 'no_gae_td': '#8993a0'}
RUN_METRICS = ['auc_return', 'final_return', 'steps_to_threshold', 'late_return_std', 'mean_approx_kl', 'max_approx_kl', 'final_policy_entropy', 'final_executed_entropy', 'final_invalid_mass', 'test_violation', 'test_replica_seconds', 'test_invalid_actions']

def run(job):
    variant, seed, steps = job
    out = ABLATION / variant
    info = train(seed, steps, out, **VARIANTS[variant])
    rows, _ = evaluate(PPO.load(out / f'policy_seed{seed}.npz'), TEST)
    test = dict(test_violation=float(np.mean([r['violation_rate'] for r in rows])), test_replica_seconds=float(np.mean([r['replica_seconds'] for r in rows])), test_invalid_actions=float(np.mean([r['invalid_actions'] for r in rows])))
    (out / f'test_seed{seed}.json').write_text(json.dumps(dict(info, **test), indent=2))
    return variant, seed

def readlog(path):
    with open(path) as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)][1:]

def per_run(variant, seed, threshold):
    log = readlog(ABLATION / variant / f'training_seed{seed}.csv')
    steps = np.array([r['steps'] for r in log])
    ret = np.array([r['mean_episode_return'] for r in log])
    reached = np.flatnonzero(ret >= threshold)
    late = ret[len(ret) // 2:]
    kl = np.array([r['approx_kl'] for r in log])
    test = json.loads((ABLATION / variant / f'test_seed{seed}.json').read_text())
    return dict(variant=variant, seed=seed, auc_return=float(ret.mean()), final_return=float(ret[-5:].mean()), steps_to_threshold=float(steps[reached[0]]) if len(reached) else float('nan'), reached_threshold=bool(len(reached)), late_return_std=float(late.std(ddof=1)), mean_approx_kl=float(kl.mean()), max_approx_kl=float(kl.max()), final_policy_entropy=log[-1]['policy_entropy'], final_executed_entropy=log[-1]['executed_entropy'], final_invalid_mass=log[-1]['invalid_mass'], test_violation=test['test_violation'], test_replica_seconds=test['test_replica_seconds'], test_invalid_actions=test['test_invalid_actions'])

def analyze(variants, seeds):
    """Convergence threshold is shared by all variants: 90% of the way from the
    median first-rollout return to the full method's mean final return."""
    logs = {v: [readlog(ABLATION / v / f'training_seed{s}.csv') for s in seeds] for v in variants}
    first = np.median([log[0]['mean_episode_return'] for v in variants for log in logs[v]])
    target = np.mean([np.mean([r['mean_episode_return'] for r in log[-5:]]) for log in logs['full']])
    threshold = float(first + 0.9 * (target - first))
    runs = [per_run(v, s, threshold) for v in variants for s in seeds]
    writecsv(ABLATION / 'runs.csv', runs)
    summary = dict(threshold_return=threshold, seeds=list(seeds), variants={})
    for v in variants:
        rr = [r for r in runs if r['variant'] == v]
        base = {r['seed']: r for r in runs if r['variant'] == 'full'}
        summary['variants'][v] = dict(label=LABELS[v], options={k: str(x) for k, x in VARIANTS[v].items()}, reached_threshold=sum(r['reached_threshold'] for r in rr), metrics={k: ci95([r[k] for r in rr]) for k in RUN_METRICS}, paired_difference_vs_full={k: ci95([r[k] - base[r['seed']][k] for r in rr]) for k in RUN_METRICS} if v != 'full' else None)
    (ABLATION / 'summary.json').write_text(json.dumps(summary, indent=2))
    write_table(summary)
    plot(logs, threshold)
    return summary

def fmt(m, scale=1, digits=2):
    if np.isnan(m['mean']):
        return 'n/a'
    if np.isnan(m['std']):
        return f"{m['mean'] * scale:.{digits}f}"
    return f"{m['mean'] * scale:.{digits}f} [{m['ci_low'] * scale:.{digits}f}, {m['ci_high'] * scale:.{digits}f}]"

def write_table(summary):
    cols = [('auc_return', 'Mean training return (AUC)', 1, 2), ('final_return', 'Final training return', 1, 2), ('steps_to_threshold', 'Decisions to threshold', 1, 0), ('late_return_std', 'Late-training return std', 1, 2), ('mean_approx_kl', 'Mean approx. KL', 1, 4), ('max_approx_kl', 'Max approx. KL', 1, 4), ('final_policy_entropy', 'Final policy entropy', 1, 3), ('final_executed_entropy', 'Final executed entropy', 1, 3), ('test_violation', 'Test violations (%)', 100, 2), ('test_replica_seconds', 'Test replica-seconds', 1, 0)]
    n = len(summary['seeds'])
    lines = [f"Means across {n} training seeds with 95% t-intervals. Threshold return = {summary['threshold_return']:.2f}; decisions-to-threshold averages only runs that reached it.", '', '| Metric | ' + ' | '.join(LABELS[v] for v in summary['variants']) + ' |', '|---|' + '---:|' * len(summary['variants'])]
    for key, name, scale, digits in cols:
        lines.append(f'| {name} | ' + ' | '.join(fmt(s['metrics'][key], scale, digits) for s in summary['variants'].values()) + ' |')
    lines.append('| Runs reaching threshold | ' + ' | '.join(f"{s['reached_threshold']}/{n}" for s in summary['variants'].values()) + ' |')
    lines += ['', 'Paired per-seed difference from full PPO (variant − full), 95% t-interval:', '', '| Metric | ' + ' | '.join(LABELS[v] for v, s in summary['variants'].items() if s['paired_difference_vs_full']) + ' |', '|---|' + '---:|' * (len(summary['variants']) - 1)]
    for key, name, scale, digits in cols:
        lines.append(f'| {name} | ' + ' | '.join(fmt(s['paired_difference_vs_full'][key], scale, digits) for s in summary['variants'].values() if s['paired_difference_vs_full']) + ' |')
    (ABLATION / 'summary.md').write_text('\n'.join(lines) + '\n')

def band(ax, logs, key, color, label, floor=-np.inf):
    steps = np.array([r['steps'] for r in logs[0]])
    ys = np.array([[r[key] for r in log] for log in logs])
    m = [ci95(col) for col in ys.T]
    ax.plot(steps, [x['mean'] for x in m], color=color, label=label)
    ax.fill_between(steps, [max(x['ci_low'], floor) for x in m], [x['ci_high'] for x in m], color=color, alpha=0.18, lw=0)

def plot(logs, threshold):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for v, vlogs in logs.items():
        band(axes[0, 0], vlogs, 'mean_episode_return', COLORS[v], LABELS[v])
        band(axes[0, 1], vlogs, 'approx_kl', COLORS[v], LABELS[v], floor=1e-4)
        band(axes[1, 0], vlogs, 'policy_entropy', COLORS[v], LABELS[v])
        band(axes[1, 1], vlogs, 'executed_entropy', COLORS[v], LABELS[v])
    axes[0, 0].axhline(threshold, color='black', ls='--', lw=1, label='Convergence threshold')
    axes[0, 0].set(title='Training return (last 20 episodes)', ylabel='Return')
    axes[0, 1].set(title='Approximate KL per update', ylabel='KL', yscale='log')
    axes[1, 0].set(title='Policy entropy (categorical, in the loss)', ylabel='nats', ylim=(0, np.log(3) * 1.05))
    axes[1, 1].set(title='Executed-action entropy (invalid → hold)', ylabel='nats', ylim=(0, np.log(3) * 1.05))
    for ax in axes.flat:
        ax.set_xlabel('Training decisions')
        ax.grid(alpha=0.15)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Component ablations: mean and 95% CI across {len(next(iter(logs.values())))} seeds")
    fig.tight_layout()
    fig.savefig(ABLATION / 'ablation_curves.png', dpi=160)
    plt.close(fig)

def main():
    args = argparse.ArgumentParser()
    args.add_argument('--steps', type=int, default=50000)
    args.add_argument('--seeds', type=int, default=10)
    args.add_argument('--jobs', type=int, default=1)
    args.add_argument('--variants', nargs='+', default=list(VARIANTS), choices=list(VARIANTS))
    args.add_argument('--analyze-only', action='store_true')
    a = args.parse_args()
    if 'full' not in a.variants:
        a.variants.insert(0, 'full')
    seeds = range(a.seeds)
    if not a.analyze_only:
        jobs = [(v, s, a.steps) for v in a.variants for s in seeds]
        with ProcessPoolExecutor(a.jobs) as pool:
            for v, s in pool.map(run, jobs):
                print(f'finished {v} seed={s}', flush=True)
    print(json.dumps(analyze(a.variants, seeds), indent=2))
if __name__ == '__main__':
    main()
