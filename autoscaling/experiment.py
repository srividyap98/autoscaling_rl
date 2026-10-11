"""Run python -m autoscaling.experiment --steps 50000 --seeds 10."""
import argparse, csv, json, time
from pathlib import Path
import numpy as np
from .environment import AutoscalingEnv, Config
from .ppo import PPO, gae, entropy_diagnostics
from .baselines import Fixed, Threshold
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)
KINDS = ['steady', 'periodic', 'bursty']
VALID = [(10000 + i, KINDS[i % 3]) for i in range(20)]
TEST = [(20000 + i, KINDS[i // 10]) for i in range(30)]

def evaluate(policy, traces, config=None, record=False):
    results = []
    histories = []
    for seed, kind in traces:
        env = AutoscalingEnv(config)
        if hasattr(policy, 'reset'):
            policy.reset()
        obs, _ = env.reset(seed, {'kind': kind, 'initial': 2})
        hist = []
        total = 0
        while True:
            action = policy.act(obs, deterministic=True)[0] if isinstance(policy, PPO) else policy.act(obs)
            obs, r, done, _, info = env.step(action)
            total += r
            if record:
                hist.append(dict(info, reward=r, action=action))
            if done:
                break
        results.append(dict(seed=seed, kind=kind, return_=total, **env.metrics()))
        histories.append(hist)
    return (results, histories)

def score(rows):
    violations = np.mean([r['violation_rate'] for r in rows])
    cost = np.mean([r['replica_seconds'] for r in rows])
    return (int(violations > 0.05), violations if violations > 0.05 else cost, cost)

def writecsv(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def train(seed, steps, out=OUT, **ppo_options):
    """Logs one row per 1,024-decision rollout; validation columns are NaN between validations."""
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    env = AutoscalingEnv()
    model = PPO(seed, **ppo_options)
    rng = np.random.default_rng(500 + seed)
    episode = 0
    obs, _ = env.reset(100000 + seed * 10000, {'kind': 'periodic'})
    logs = []
    episode_return = 0
    completed = []
    bestscore = None
    beststep = 0

    def validate(step):
        nonlocal bestscore, beststep
        rows, _ = evaluate(model, VALID)
        s = score(rows)
        if step > 0 and (bestscore is None or s <= bestscore):
            bestscore = s
            beststep = step
            model.save(out / f'policy_seed{seed}.npz')
        return (float(np.mean([r['violation_rate'] for r in rows])), float(np.mean([r['replica_seconds'] for r in rows])))
    v, c = validate(0)
    nan = float('nan')
    logs.append(dict(seed=seed, steps=0, mean_episode_return=nan, validation_violation=v, validation_cost=c, elapsed_seconds=0, entropy=nan, actor_loss=nan, value_mse=nan, clip_fraction=nan, approx_kl=nan, policy_entropy=nan, executed_entropy=nan, invalid_mass=nan))
    step = 0
    while step < steps:
        states = []
        valid = []
        actions = []
        oldlogs = []
        values = []
        rewards = []
        nextvalues = []
        terminals = []
        for _ in range(min(1024, steps - step)):
            a, lp, value = model.act(obs)
            valid.append(env.valid_actions())
            nxt, r, done, _, info = env.step(a)
            states.append(obs)
            actions.append(a)
            oldlogs.append(lp)
            values.append(value)
            rewards.append(r)
            nextvalues.append(0 if done else model.value(nxt))
            terminals.append(done)
            obs = nxt
            step += 1
            episode_return += r
            if done:
                completed.append(episode_return)
                episode_return = 0
                episode += 1
                obs, _ = env.reset(100000 + seed * 10000 + episode, {'kind': str(rng.choice(['periodic', 'bursty']))})
        states = np.array(states)
        rollout = entropy_diagnostics(model.probabilities(states), np.array(valid))
        advantage, returns = gae(np.array(rewards), np.array(values), np.array(nextvalues), np.array(terminals), model.gamma, model.lam)
        stats = model.update(states, np.array(actions), np.array(oldlogs), advantage, returns)
        validated = step % 10240 == 0 or step == steps
        v, c = validate(step) if validated else (nan, nan)
        row = dict(seed=seed, steps=step, mean_episode_return=float(np.mean(completed[-20:])) if completed else nan, validation_violation=v, validation_cost=c, elapsed_seconds=time.perf_counter() - start, **stats, **rollout)
        logs.append(row)
        if validated:
            writecsv(out / f'training_seed{seed}.csv', logs)
            print(f"{out.name} seed={seed} steps={step} return={row['mean_episode_return']:.2f} validation violations={v:.4%} cost={c:.0f} entropy policy={rollout['policy_entropy']:.3f} executed={rollout['executed_entropy']:.3f}", flush=True)
    model.save(out / f'final_seed{seed}.npz')
    return dict(seed=seed, steps=step, best_step=beststep, elapsed_seconds=time.perf_counter() - start)

def main():
    args = argparse.ArgumentParser()
    args.add_argument('--steps', type=int, default=50000)
    args.add_argument('--seeds', type=int, default=10)
    a = args.parse_args()
    fixeds = []
    for n in range(1, 11):
        rows, _ = evaluate(Fixed(n), VALID)
        fixeds.append((score(rows), n))
    fixed_n = min(fixeds)[1]
    choices = []
    for upper in [0.6, 0.7, 0.8]:
        for lower in [0.2, 0.3, 0.4]:
            for cooldown in [10, 20]:
                rows, _ = evaluate(Threshold(upper, lower, cooldown), VALID)
                choices.append((score(rows), upper, lower, cooldown))
    _, upper, lower, cooldown = min(choices)
    print('Validated baselines:', fixed_n, upper, lower, cooldown, flush=True)
    meta = dict(config=Config().__dict__, fixed_replicas=fixed_n, threshold=dict(upper=upper, lower=lower, cooldown=cooldown), validation=VALID, test=TEST, runs=[], framework='NumPy explicit backpropagation')
    rows = []
    for name, policy in [('fixed', Fixed(fixed_n)), ('threshold', Threshold(upper, lower, cooldown))]:
        rr, _ = evaluate(policy, TEST)
        rows.extend((dict(policy=name, training_seed=-1, **r) for r in rr))
    for seed in range(a.seeds):
        meta['runs'].append(train(seed, a.steps))
        rr, _ = evaluate(PPO.load(OUT / f'policy_seed{seed}.npz'), TEST)
        rows.extend((dict(policy='ppo', training_seed=seed, **r) for r in rr))
        writecsv(OUT / 'test_metrics.csv', rows)
        (OUT / 'experiment.json').write_text(json.dumps(meta, indent=2))
    ranking = []
    for seed in range(a.seeds):
        rr, _ = evaluate(PPO.load(OUT / f'policy_seed{seed}.npz'), VALID)
        ranking.append((score(rr), seed))
    meta['demo_seed'] = min(ranking)[1]
    (OUT / 'experiment.json').write_text(json.dumps(meta, indent=2))
    print('Experiment complete', flush=True)
if __name__ == '__main__':
    main()
