# PPO for simulated API autoscaling

A course project implementing clipped categorical PPO and Generalized Advantage Estimation from scratch in NumPy. This repository includes the simulator, tests, five trained actor–critic policies, raw experiment logs, metrics, a demo, and a presentation. Read `outputs/summary.json` for the actual outcomes. All workload and capacity values are simulation assumptions.

## Measured outcome

| Controller | Mean replica-seconds | Mean violation rate |
|---|---:|---:|
| Fixed target 8 | 14,250.00 | 4.58% |
| Threshold | 15,905.00 | 6.59% |
| PPO (5 policies) | 17,599.67 | 2.50% |

PPO used 23.51% more capacity than fixed. Its steady-traffic violation mean was 5.05%, so both the cost target and the per-workload service requirement failed. Training returns improved descriptively, but greedy validation could already look favorable at the untrained checkpoint. These experiments do not establish that training caused better deterministic service behavior. PPO frequently asks to add replicas at the limit (about 172 invalid actions per 180-decision test episode), behaving largely like a high-capacity fixed controller. This is a completed experiment with a failed operational hypothesis, not a cost-saving deployment result.

## Run locally

Python 3.10 or later:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m autoscaling.experiment --steps 50000 --seeds 5
python -m autoscaling.analyze
```

The analysis script creates plots and `outputs/agent_demo.mp4`. Video generation additionally needs the **FFmpeg executable** on PATH. The demo already included in this repository can be viewed without FFmpeg. Training itself needs only NumPy. To obtain the measured reference runtime, use `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` before the training command on macOS/Linux. Hardware and NumPy builds affect runtime.

## Files to read first

1. `autoscaling/environment.py`: arrivals, FIFO queue, startup delay, scaling limits, reward, and request accounting.
2. `autoscaling/ppo.py`: separate actor/critic networks, clipped objective, entropy gradient, GAE, manual backpropagation, and Adam.
3. `autoscaling/experiment.py`: validation-only tuning, repeated training, checkpoint selection, and identical test arrivals.
4. `autoscaling/baselines.py`: a constant target replica count and a threshold controller.
5. `autoscaling/analyze.py`: summary statistics, plots, and a replay of real evaluated policy decisions.
6. `tests/test_core.py`: finite-difference gradient checks, terminal handling, startup timing, accounting, and seeded reproducibility.

## Environment and metrics

Each episode has 1,800 simulated seconds and one decision every ten seconds. All controllers start with two active replicas. A replica serves up to 20 requests per second and starts after 20 seconds. Capacity cannot fall below one active replica or exceed ten allocated replicas. Actions 0/1/2 mean remove/hold/add. Invalid actions are no-ops and logged separately from successful changes.

The nine observations are recent arrival rate, a three-interval rate average, rate change, queue length, oldest waiting age, active replicas, two startup-stage counts, and utilization. These summaries do not reveal the full simulator state or future demand. The queue is FIFO, stores arrival cohorts, holds at most 1,000 requests, and expires requests after ten seconds. Serving after two seconds, expiration, and overflow are all waiting-target violations. Queue delay excludes application processing and network transit. Poisson arrivals use predetermined rates: steady 35–150, periodic roughly 20–160, and bursts 130–180 requests/second. These distributions are chosen examples, not measured production traces.

Reward is negative cost: 0.1 × mean allocated replicas / 10; 5 × the violation fraction among requests resolved that interval; 0.5 × mean queue / 1,000; and 0.02 per executed scaling action. Starting replicas incur cost. A zero-resolution interval has zero violation fraction. After arrivals stop, capacity freezes and the queue drains, with drain cost and failures included. Because this task ends after drainage, the implementation uses true terminal episodes and does not bootstrap across them.

The actor is a categorical 9–64–64–3 network with tanh hidden layers. The critic is a separate 9–64–64–1 network. Adam uses learning rate 0.0003, discount 0.99, GAE lambda 0.95, clip 0.2, entropy coefficient 0.01, value coefficient 0.5, rollout size 1,024, ten epochs, minibatches up to 64, and gradient norm limit 0.5. Float64 aids finite-difference checking. All rollouts are on-policy and action sampling uses seeded generators. Evaluation uses the highest-probability action.

## Experimental protocol

- Five independent training seeds, 50,000 decisions per seed.
- Twenty validation traces choose checkpoints and baseline settings. An untrained checkpoint is logged but cannot be selected as a submitted learned policy. Ties prefer later trained checkpoints.
- Thirty test traces, ten each steady, periodic, and bursty. Test seeds are separate from validation and training.
- The fixed baseline ramps from two to its validated constant target, then holds. It never receives instantly available extra capacity.
- Threshold upper values 0.6/0.7/0.8, lower values 0.2/0.3/0.4, and cooldown values 10/20 seconds are searched on validation. Low utilization must persist for three decisions with an empty queue before downscaling.
- Summary values are arithmetic means of per-episode metrics, not a request-weighted aggregate. The mean of episode p95 values is not a pooled p95.
- PPO has 150 test rows (five policies × thirty traces). Baselines each have thirty rows. These are repeated policies on common traces, not 150 independently generated workloads.
- No cost savings or deployment claims are inferred from simulated reward alone.

## Changes from Part 1

PyTorch/Gymnasium were unavailable in the execution workspace. The completed implementation uses NumPy and Gymnasium-style reset/step signatures without registering a Gymnasium environment. Architecture, clipped PPO, GAE, and five 50,000-decision runs follow the plan. Numerical gradient tests support the custom implementation. The initial reward weights remain fixed rather than receiving an extensive search. Baseline settings and checkpoints use validation only. A finite workload followed by queue drainage is modeled as terminal, rather than a truncation of continuing traffic. DQN remains a discussed alternative; it is not implemented.

## Demonstration and recording

`outputs/agent_demo.mp4` is a silent 60-second replay of actual decisions on predetermined steady, periodic, and bursty test traces. Full curves remain visible; a cursor shows replay progress. Each 20-second segment represents 30 simulated minutes. It is an illustration, not a substitute for the metrics across all traces. The demo policy is selected using validation performance only.

The course requires a **13–15 minute recording with your own face and voice throughout**. Use the deck and `presentation/Narration_and_Recording_Guide.md`, keep your webcam visible, and narrate while playing the demo. You must create that final recording yourself. The prepared demo contains no fabricated webcam or narration.

## Repository submission

This folder is a complete local Git repository. To provide an accessible repository URL, create a GitHub repository under your account, upload this folder or push the included commit, and verify the instructor has access. Do not submit a local path as a repository URL. The ZIP is a portable copy, not evidence that a remote GitHub repository has been published.

## AI use disclosure

OpenAI ChatGPT helped implement code, write tests, organize the experiment, analyze measured outputs, and draft slides and narration. Training metrics come from actual executions of the included simulator. 

## Research sources

- Schulman et al. (2017), *Proximal Policy Optimization Algorithms*. https://arxiv.org/abs/1707.06347
- Schulman et al. (2015), *High-Dimensional Continuous Control Using Generalized Advantage Estimation*. https://arxiv.org/abs/1506.02438
- Barrett et al. (2013), *Applying reinforcement learning towards automating resource allocation and application scalability in the cloud*. https://doi.org/10.1002/cpe.2864
- Garí et al. (2020), *Reinforcement Learning-based Application Autoscaling in the Cloud: A Survey*. https://arxiv.org/abs/2001.09957
- Wang et al. (2017), *Automated Cloud Provisioning on AWS using Deep Reinforcement Learning*. https://arxiv.org/abs/1709.04305
- Qiu et al. (2023), *AWARE: Automate Workload Autoscaling with Reinforcement Learning in Production Cloud Systems*. https://www.usenix.org/conference/atc23/presentation/qiu-haoran
- Mnih et al. (2013), *Playing Atari with Deep Reinforcement Learning*. https://arxiv.org/abs/1312.5602
- Henderson et al. (2017), *Deep Reinforcement Learning that Matters*. https://arxiv.org/abs/1709.06560
