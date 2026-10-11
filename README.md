# PPO for simulated API autoscaling

A course project implementing clipped categorical PPO and Generalized Advantage Estimation from scratch in NumPy. This repository includes the simulator, tests, ten trained actor–critic policies, component ablations, raw experiment logs, metrics, a demo, and a presentation. Read `outputs/summary.json` for the actual outcomes. All workload and capacity values are simulation assumptions.

## Measured outcome

| Controller | Mean replica-seconds | Mean violation rate |
|---|---:|---:|
| Fixed target 8 | 14,250.00 | 4.58% |
| Threshold | 15,905.00 | 6.59% |
| PPO (10 seeds, mean [95% CI across seeds]) | 17,648.84 [17,521.82, 17,775.85] | 2.21% [1.47%, 2.94%] |

PPO used 23.85% more capacity than fixed. Its steady-traffic violation mean was 4.18% [2.05%, 6.31%]; one seed (seed 2) reached 5.11% overall, and the steady-traffic interval crosses the 5% limit, so neither the cost target nor the per-workload service requirement is met with confidence. Training returns improved descriptively, but greedy validation could already look favorable at the untrained checkpoint. These experiments do not establish that training caused better deterministic service behavior. PPO frequently asks to add replicas at the limit (167 [157, 178] invalid actions per 180-decision test episode), behaving largely like a high-capacity fixed controller. The entropy diagnostics below show this directly: by the end of training about 63% of the policy's probability goes to actions the environment rejects. This is a completed experiment with a failed operational hypothesis, not a cost-saving deployment result.

Seeds 0–4 reproduce the original five-seed checkpoints byte for byte; seeds 5–9 were added for the confidence intervals.

## Run locally

Python 3.10 or later:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m autoscaling.experiment --steps 50000 --seeds 10
python -m autoscaling.analyze
python -m autoscaling.ablation --steps 50000 --seeds 10 --jobs 4
```

The ablation command trains 4 variants × 10 seeds (about 10 minutes with four worker processes) and writes `outputs/ablations/`. Add `--analyze-only` to rebuild its tables and plots from existing logs.

The analysis script creates plots and `outputs/agent_demo.mp4`. Video generation additionally needs the **FFmpeg executable** on PATH. The demo already included in this repository can be viewed without FFmpeg. Training itself needs only NumPy. To obtain the measured reference runtime, use `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` before the training command on macOS/Linux. Hardware and NumPy builds affect runtime.

## Files to read first

1. `autoscaling/environment.py`: arrivals, FIFO queue, startup delay, scaling limits, reward, and request accounting.
2. `autoscaling/ppo.py`: separate actor/critic networks, clipped objective, entropy gradient, GAE, manual backpropagation, and Adam.
3. `autoscaling/experiment.py`: validation-only tuning, repeated training, checkpoint selection, and identical test arrivals.
4. `autoscaling/baselines.py`: a constant target replica count and a threshold controller.
5. `autoscaling/analyze.py`: summary statistics, seed-level confidence intervals, plots, and a replay of real evaluated policy decisions.
6. `autoscaling/ablation.py`: paired-seed ablations of clipping and GAE, with convergence and stability metrics.
7. `autoscaling/stats.py`: Student-t 95% confidence intervals across training seeds.
8. `tests/test_core.py`: finite-difference gradient checks (clipped and unclipped), GAE terminal handling and λ extremes, executed-action entropy, action masks, startup timing, accounting, and seeded reproducibility.

## Environment and metrics

Each episode has 1,800 simulated seconds and one decision every ten seconds. All controllers start with two active replicas. A replica serves up to 20 requests per second and starts after 20 seconds. Capacity cannot fall below one active replica or exceed ten allocated replicas. Actions 0/1/2 mean remove/hold/add. Invalid actions are no-ops and logged separately from successful changes.

The nine observations are recent arrival rate, a three-interval rate average, rate change, queue length, oldest waiting age, active replicas, two startup-stage counts, and utilization. These summaries do not reveal the full simulator state or future demand. The queue is FIFO, stores arrival cohorts, holds at most 1,000 requests, and expires requests after ten seconds. Serving after two seconds, expiration, and overflow are all waiting-target violations. Queue delay excludes application processing and network transit. Poisson arrivals use predetermined rates: steady 35–150, periodic roughly 20–160, and bursts 130–180 requests/second. These distributions are chosen examples, not measured production traces.

Reward is negative cost: 0.1 × mean allocated replicas / 10; 5 × the violation fraction among requests resolved that interval; 0.5 × mean queue / 1,000; and 0.02 per executed scaling action. Starting replicas incur cost. A zero-resolution interval has zero violation fraction. After arrivals stop, capacity freezes and the queue drains, with drain cost and failures included. Because this task ends after drainage, the implementation uses true terminal episodes and does not bootstrap across them.

The actor is a categorical 9–64–64–3 network with tanh hidden layers. The critic is a separate 9–64–64–1 network. Adam uses learning rate 0.0003, discount 0.99, GAE lambda 0.95, clip 0.2, entropy coefficient 0.01, value coefficient 0.5, rollout size 1,024, ten epochs, minibatches up to 64, and gradient norm limit 0.5. Float64 aids finite-difference checking. All rollouts are on-policy and action sampling uses seeded generators. Evaluation uses the highest-probability action.

## Experimental protocol

- Ten independent training seeds, 50,000 decisions per seed. Training logs one row per 1,024-decision rollout (`outputs/training_seed*.csv`); validation columns are filled every 10,240 decisions.
- Twenty validation traces choose checkpoints and baseline settings. An untrained checkpoint is logged but cannot be selected as a submitted learned policy. Ties prefer later trained checkpoints.
- Thirty test traces, ten each steady, periodic, and bursty. Test seeds are separate from validation and training.
- The fixed baseline ramps from two to its validated constant target, then holds. It never receives instantly available extra capacity.
- Threshold upper values 0.6/0.7/0.8, lower values 0.2/0.3/0.4, and cooldown values 10/20 seconds are searched on validation. Low utilization must persist for three decisions with an empty queue before downscaling.
- Summary values are arithmetic means of per-episode metrics, not a request-weighted aggregate. The mean of episode p95 values is not a pooled p95.
- PPO has 300 test rows (ten policies × thirty traces). Baselines each have thirty rows. These are repeated policies on common traces, not 300 independently generated workloads. Confidence intervals therefore treat the **training seed** as the unit: each seed's 30 test episodes are averaged first, and the 95% Student-t interval is formed across the ten seed means (`ppo_seed_level_95ci` in `outputs/summary.json`). The intervals describe training-seed variability on these fixed test traces, not uncertainty over workloads. The baselines are deterministic on these traces, so they have no seed interval.
- No cost savings or deployment claims are inferred from simulated reward alone.

## Component ablations

`autoscaling/ablation.py` retrains four variants on the same ten seeds. Seed *s* gives every variant identical initial weights and training traffic, so each variant is compared with full PPO through paired per-seed differences. Checkpoint selection and test traces are the same as in the main experiment, and the `full` variant reproduces it exactly.

| Variant | Change |
|---|---|
| Full PPO | clip 0.2, GAE λ = 0.95 |
| No clipping | clip = ∞: an unclipped importance-weighted surrogate, still 10 epochs × minibatches |
| No GAE, Monte Carlo | λ = 1: discounted return minus the value baseline |
| No GAE, one-step TD | λ = 0: r + γV(s′) − V(s) |

Convergence speed is the number of decisions until the 20-episode training return first reaches a threshold shared by all variants: 90% of the way from the median first-rollout return to full PPO's mean final return (−82.33). Stability is measured as the approximate KL per update, the standard deviation of training return over the second half of training, and spread across seeds. Means are shown with 95% intervals across 10 seeds; paired differences are variant − full.

| Metric | Full PPO | No clipping | No GAE (λ=1) | No GAE (λ=0) |
|---|---:|---:|---:|---:|
| Decisions to threshold | 6,451 [5,848, 7,054] | 5,427 [4,732, 6,122] | 6,656 [6,270, 7,042] | 9,318 [7,509, 11,127] |
| Paired difference vs full | | −1,024 [−1,512, −536] | +205 [−258, +668] | +2,867 [+951, +4,783] |
| Mean training return (AUC) | −66.38 [−69.48, −63.29] | −58.81 [−61.57, −56.05] | −68.28 [−71.36, −65.20] | −90.96 [−102.25, −79.67] |
| Final training return | −36.73 [−38.95, −34.50] | −35.50 [−38.66, −32.34] | −36.31 [−38.39, −34.23] | −41.67 [−54.51, −28.84] |
| Late-training return std | 5.93 [4.22, 7.64] | 8.29 [4.50, 12.09] | 5.21 [4.01, 6.41] | 13.39 [5.71, 21.07] |
| Mean approx. KL per update | 0.0043 | 0.0583 (13.5×) | 0.0041 | 0.0066 |
| Max approx. KL in a run | 0.015 [0.014, 0.017] | 0.600 [0.299, 0.901] | 0.013 [0.012, 0.014] | 0.030 [0.026, 0.034] |
| Test violations | 2.21% [1.47, 2.94] | 2.09% [1.56, 2.61] | 2.19% [1.44, 2.94] | 6.36% [−0.88, 13.60] |
| Test replica-seconds | 17,649 | 17,402 | 17,650 | 17,044 |

The full tables, including every paired difference, are in `outputs/ablations/summary.md`; per-run values are in `runs.csv`; curves are in `ablation_curves.png`.

What the ablations show, within this simulator and budget:

- **Clipping trades speed for stability.** Without clipping, training reaches the threshold about 1,000 decisions sooner and has a higher average training return. But per-update KL rises about 13-fold, and single updates reach KL ≈ 0.6, against about 0.015 with clipping. Late-training return is noisier, though that interval overlaps full PPO's. One unclipped run (seed 0) ended training in a collapsed state, failing 94% of validation requests at minimum capacity. Checkpoint selection on validation hides this from the test metrics, which do not differ significantly.
- **GAE's contribution depends on what replaces it.** With λ = 1 (Monte Carlo), results are statistically indistinguishable from λ = 0.95 except for a slightly lower average training return (paired −1.90 [−3.11, −0.68]). Episodes are short and terminal, so the extra variance of full returns costs little here. With λ = 0 (one-step TD), the biased advantage slows convergence by about 2,900 decisions, lowers the average return by 24.6, and makes training the least stable. One λ = 0 run (seed 1) collapsed late in training, and the test-violation interval is the widest.
- **No variant changes the operational conclusion.** Every variant converges to a high-capacity policy of roughly 17,000–17,650 replica-seconds, well above fixed's 14,250.

## Policy entropy vs executed-action entropy

The actor is **categorical** over remove/hold/add, not a squashed Gaussian, so there is no change-of-variables correction to apply. Its entropy is computed exactly. The environment does transform actions, though: remove at one active replica and add at ten allocated replicas are executed as hold. Each rollout therefore logs three quantities on the states actually visited (columns in `training_seed*.csv`, plotted in `outputs/entropy.png`):

- `policy_entropy`: entropy of the softmax policy, the distribution the entropy bonus acts on.
- `executed_entropy`: entropy after pushing that distribution through the environment's masking, so that invalid probability moves to hold.
- `invalid_mass`: probability the policy places on actions the environment will reject.

The column `entropy` is unchanged from before: the policy entropy averaged over minibatches during the update.

At the end of training, full PPO has a policy entropy of 0.645 nats but an executed entropy of only 0.169 nats, with 63% of probability on invalid actions. The policy looks moderately exploratory to the loss, but most of that randomness is spent choosing between add and hold at the replica ceiling, and both mean hold. The 0.01 entropy bonus is therefore keeping exploration alive mostly in actions with no effect. This explains the large invalid-action counts. It also suggests that action masking, or an entropy bonus on the executed distribution, is a natural next experiment.

## Changes from Part 1

PyTorch/Gymnasium were unavailable in the execution workspace. The completed implementation uses NumPy and Gymnasium-style reset/step signatures without registering a Gymnasium environment. Architecture, clipped PPO, GAE, and the planned five 50,000-decision runs (since extended to ten seeds plus ablations) follow the plan. Numerical gradient tests support the custom implementation. The initial reward weights remain fixed rather than receiving an extensive search. Baseline settings and checkpoints use validation only. A finite workload followed by queue drainage is modeled as terminal, rather than a truncation of continuing traffic. DQN remains a discussed alternative; it is not implemented.

## Demonstration and recording

`outputs/agent_demo.mp4` is a silent 60-second replay (validation-selected seed 1) of actual decisions on predetermined steady, periodic, and bursty test traces. Full curves remain visible; a cursor shows replay progress. Each 20-second segment represents 30 simulated minutes. It is an illustration, not a substitute for the metrics across all traces. The demo policy is selected using validation performance only.

The course requires a **13–15 minute recording with your own face and voice throughout**. Use the deck and `presentation/Narration_and_Recording_Guide.md`, keep your webcam visible, and narrate while playing the demo. You must create that final recording yourself. The prepared demo contains no fabricated webcam or narration.

## Repository submission

The repository is published at https://github.com/srividyap98/autoscaling_rl . Verify the instructor has access. Do not submit a local path as a repository URL.

## AI use disclosure

OpenAI ChatGPT helped implement code, write tests, organize the experiment, analyze measured outputs, and draft slides and narration. Anthropic Claude helped add the ten-seed confidence intervals, component ablations, and entropy diagnostics, and update the README, slides, and narration to match. Training metrics come from actual executions of the included simulator.

## Research sources

- Schulman et al. (2017), *Proximal Policy Optimization Algorithms*. https://arxiv.org/abs/1707.06347
- Schulman et al. (2015), *High-Dimensional Continuous Control Using Generalized Advantage Estimation*. https://arxiv.org/abs/1506.02438
- Barrett et al. (2013), *Applying reinforcement learning towards automating resource allocation and application scalability in the cloud*. https://doi.org/10.1002/cpe.2864
- Garí et al. (2020), *Reinforcement Learning-based Application Autoscaling in the Cloud: A Survey*. https://arxiv.org/abs/2001.09957
- Wang et al. (2017), *Automated Cloud Provisioning on AWS using Deep Reinforcement Learning*. https://arxiv.org/abs/1709.04305
- Qiu et al. (2023), *AWARE: Automate Workload Autoscaling with Reinforcement Learning in Production Cloud Systems*. https://www.usenix.org/conference/atc23/presentation/qiu-haoran
- Mnih et al. (2013), *Playing Atari with Deep Reinforcement Learning*. https://arxiv.org/abs/1312.5602
- Henderson et al. (2017), *Deep Reinforcement Learning that Matters*. https://arxiv.org/abs/1709.06560
