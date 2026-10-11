"""Aggregate raw results and make scientific plots and a real agent demo."""
import csv, json, platform, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from .experiment import OUT, evaluate, VALID, TEST, score, writecsv
from .environment import Config
from .baselines import Fixed, Threshold
from .ppo import PPO

def readcsv(path):
    with open(path) as f:
        return [{k: v if k in ['policy', 'kind'] else float(v) for k, v in r.items()} for r in csv.DictReader(f)]

def main():
    meta = json.loads((OUT / 'experiment.json').read_text())
    fixeds = []
    for n in range(1, 11):
        rr, _ = evaluate(Fixed(n), VALID)
        fixeds.append((score(rr), n))
    meta['fixed_replicas'] = min(fixeds)[1]
    choices = []
    for upper in [0.6, 0.7, 0.8]:
        for lower in [0.2, 0.3, 0.4]:
            for cooldown in [10, 20]:
                rr, _ = evaluate(Threshold(upper, lower, cooldown), VALID)
                choices.append((score(rr), upper, lower, cooldown))
    _, u, l, c = min(choices)
    meta['threshold'] = dict(upper=u, lower=l, cooldown=c)
    fixed = Fixed(meta['fixed_replicas'])
    threshold = Threshold(u, l, c)
    rows = []
    for name, policy in [('fixed', fixed), ('threshold', threshold)]:
        rr, _ = evaluate(policy, TEST)
        rows.extend((dict(policy=name, training_seed=-1, **r) for r in rr))
    for run in meta['runs']:
        rr, _ = evaluate(PPO.load(OUT / f"policy_seed{run['seed']}.npz"), TEST)
        rows.extend((dict(policy='ppo', training_seed=run['seed'], **r) for r in rr))
    writecsv(OUT / 'test_metrics.csv', rows)
    summary = {}
    for name in ['fixed', 'threshold', 'ppo']:
        summary[name] = {}
        for kind in ['all', 'steady', 'periodic', 'bursty']:
            rr = [r for r in rows if r['policy'] == name and (kind == 'all' or r['kind'] == kind)]
            summary[name][kind] = {k: float(np.mean([r[k] for r in rr])) for k in ['replica_seconds', 'violation_rate', 'p95_wait_seconds', 'scaling_actions', 'invalid_actions', 'max_queue', 'return_']}
    per_run = []
    for run in meta['runs']:
        rr = [r for r in rows if r['policy'] == 'ppo' and r['training_seed'] == run['seed']]
        per_run.append(dict(seed=run['seed'], cost=float(np.mean([r['replica_seconds'] for r in rr])), violation=float(np.mean([r['violation_rate'] for r in rr]))))
    summary['ppo_runs'] = per_run
    summary['cost_saving_vs_fixed'] = 1 - summary['ppo']['all']['replica_seconds'] / summary['fixed']['all']['replica_seconds']
    summary['goal_met'] = bool(summary['cost_saving_vs_fixed'] >= 0.1 and all((summary['ppo'][k]['violation_rate'] <= 0.05 for k in ['steady', 'periodic', 'bursty'])))
    summary['training_seconds'] = float(sum((r['elapsed_seconds'] for r in meta['runs'])))
    summary['machine'] = {'platform': platform.platform(), 'processor': platform.processor(), 'numpy': np.__version__, 'python': platform.python_version()}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
    (OUT / 'experiment.json').write_text(json.dumps(meta, indent=2))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = {'fixed': '#8993a0', 'threshold': '#21867a', 'ppo': '#dc7841'}
    fig, ax = plt.subplots(figsize=(10, 5))
    for run in meta['runs']:
        log = readcsv(OUT / f"training_seed{run['seed']}.csv")
        ax.plot([x['steps'] for x in log[1:]], [x['mean_episode_return'] for x in log[1:]], label=f"Seed {run['seed']}")
    ax.set(xlabel='Training decisions', ylabel='Mean return over last 20 episodes', title='Training return varies across independent runs')
    ax.legend(ncol=5)
    fig.tight_layout()
    fig.savefig(OUT / 'learning_curve.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    names = ['fixed', 'threshold', 'ppo']
    for ax, key, title, scale in [(axes[0], 'replica_seconds', 'Allocated replica-seconds', 1), (axes[1], 'violation_rate', 'Requests missing waiting target (%)', 100)]:
        ax.bar(names, [summary[n]['all'][key] * scale for n in names], color=[colors[n] for n in names])
        ax.set_title(title)
        if scale == 100:
            ax.axhline(5, color='black', ls='--', lw=1, label='5% limit')
            ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / 'comparison.png', dpi=160)
    plt.close(fig)
    policy = PPO.load(OUT / f"policy_seed{meta['demo_seed']}.npz")
    demo_data = []
    for seed, kind in [(20000, 'steady'), (20010, 'periodic'), (20020, 'bursty')]:
        metrics, h = evaluate(policy, [(seed, kind)], record=True)
        demo_data.append(dict(seed=seed, kind=kind, metrics=metrics[0], history=h[0]))
    (OUT / 'demo_trace.json').write_text(json.dumps(demo_data, indent=2))
    fig, axes = plt.subplots(3, 1, figsize=(12.8, 7.2), sharex=True)
    fig.subplots_adjust(hspace=0.28, left=0.09, right=0.96, top=0.86, bottom=0.1)
    writer = FFMpegWriter(fps=9, codec='libx264', extra_args=['-pix_fmt', 'yuv420p', '-crf', '22'])
    with writer.saving(fig, str(OUT / 'agent_demo.mp4'), 100):
        for demo in demo_data:
            h = demo['history']
            x = np.array([r['time'] / 60 for r in h])
            rate = [r['rate'] for r in h]
            capacity = [20 * r['active'] for r in h]
            queue = [r['queue'] for r in h]
            bad = np.cumsum([r['interval_bad'] for r in h])
            resolved = np.cumsum([r['interval_resolved'] for r in h])
            violation = 100 * bad / np.maximum(resolved, 1)
            for ax in axes:
                ax.clear()
                ax.set_xlim(0, 30)
                ax.grid(alpha=0.15)
            axes[0].plot(x, rate, color='#8c96a5', label='Observed demand')
            axes[0].plot(x, capacity, color='#dc7841', label='PPO active capacity')
            axes[0].set_ylim(0, 220)
            axes[0].set_ylabel('Requests / second')
            axes[0].legend(loc='upper right')
            axes[1].plot(x, queue, color='#21867a')
            axes[1].set_ylim(0, 1000)
            axes[1].set_ylabel('Queued requests')
            axes[2].plot(x, violation, color='#dc7841')
            axes[2].axhline(5, color='black', ls='--')
            axes[2].set_ylim(0, max(10, float(max(violation)) + 2))
            axes[2].set_ylabel('Resolved violations (%)')
            axes[2].set_xlabel('Simulated minutes')
            cursors = [ax.axvline(0, color='black', lw=1.8) for ax in axes]
            title = fig.suptitle('', fontsize=17)
            for i, r in enumerate(h):
                for line in cursors:
                    line.set_xdata([x[i], x[i]])
                title.set_text(f"PPO replay: {demo['kind']} traffic | {x[i]:.1f} min | {r['active']} active, {r['pending']} starting | queue {r['queue']}")
                writer.grab_frame()
            fig.savefig(OUT / f"demo_{demo['kind']}.png", dpi=130)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
if __name__ == '__main__':
    main()
