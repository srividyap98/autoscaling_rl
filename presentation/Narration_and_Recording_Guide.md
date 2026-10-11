# Narration and recording guide

This is a draft for you to review, understand, and personalize. The target is **13 minutes 55 seconds**, including a narrated 60-second demo. The approximately 1,800-word script requires about 129 words per minute overall, which leaves room for pauses. Practice once and adjust to your natural pace; the timing plan is not proof of a 13–15 minute recording.

## Before recording

1. Open `presentation/API_Autoscaling_RL_Presentation.pptx` and `outputs/agent_demo.mp4`. The demo is separate from the deck; preload it in a video player and check the switch from slide 15 to the player and back.
2. Use a screen recorder that captures your slides, webcam, and microphone together. ScreenPal supports this combination: https://screenpal.com/tutorial/record-your-screen-and-webcam . Keep your face visible throughout, including during the demo. Position your webcam overlay where it does not cover chart labels; move it if needed.
3. Keep speaker notes or this guide on a second screen or printout. Record a short sample to check legibility and audio before recording the full presentation.
4. Review the code and sources, explain the outcome in your own words, and retain an accurate AI disclosure. Add your name/course details if required by your instructor.
5. Verify the final recording lasts 13–15 minutes, has your face and voice throughout, includes all three demo segments, and shows slides clearly.

## Timing plan

| Slide | Topic | Time | Cumulative end |
|---|---|---:|---:|
| 1 | Learning to scale an API | 35 s | 0:35 |
| 2 | The goal from Part 1 | 50 s | 1:25 |
| 3 | Research that shaped the design | 40 s | 2:05 |
| 4 | A small, explicit simulator | 50 s | 2:55 |
| 5 | What the agent sees and controls | 40 s | 3:35 |
| 6 | Reward expresses the tradeoff | 45 s | 4:20 |
| 7 | PPO and the actor–critic model | 55 s | 5:15 |
| 8 | Implementation and changes to the plan | 45 s | 6:00 |
| 9 | Training and evaluation protocol | 45 s | 6:45 |
| 10 | Training improved, but runs differed | 45 s | 7:30 |
| 11 | Better average service, higher cost | 60 s | 8:30 |
| 12 | Workload and seed results matter | 50 s | 9:20 |
| 13 | Ablations: what clipping and GAE contribute | 55 s | 10:15 |
| 14 | The policy's entropy hides inactive choices | 45 s | 11:00 |
| 15 | Demonstration: actual policy replay | 65 s | 12:05 |
| 16 | Reliability and ethical considerations | 45 s | 12:50 |
| 17 | What I learned and what comes next | 40 s | 13:30 |
| 18 | References and AI disclosure | 25 s | 13:55 |

## Demo cue sheet

At slide 15, introduce the replay for about five seconds, then switch to the preloaded video. Continue speaking for its full minute. Keep your webcam overlay active. Return to slide 16 after playback.

| Video time | What to discuss |
|---|---|
| 0–20 s | Steady traffic: initial ramp, excess capacity, queue behavior. |
| 20–40 s | Periodic traffic: demand falls while capacity stays high. |
| 40–60 s | Bursts: spare capacity absorbs peaks; this does not prove efficient scaling. |

The full curves remain visible and the cursor indicates replay time. Orange is active capacity/demand panels and violations; gray is observed demand; green is queue. The cumulative violation curve uses resolved requests, while the final episode metric includes all arrivals after queue drainage. The replay uses validation-selected seed 1 and test seeds 20000, 20010, and 20020. It is not a live cloud demo.

## Slide-by-slide draft narration

### Slide 1: Learning to scale an API

Target: 35 seconds.

This project asks whether a reinforcement learning agent can choose how many API replicas to run as demand changes. Too little capacity makes requests wait, while too much capacity wastes resources. Following my Part 1 plan, I built a local simulator, trained PPO across ten random seeds, compared it with two simpler controllers, and ran ablations of PPO's main components. The main finding is that the agent protected service better on average, but did not achieve the planned cost savings.

### Slide 2: The goal from Part 1

Target: 50 seconds.

The original objective had two parts. First, at least 95 percent of requests should meet a two-second waiting target. Second, PPO should use at least ten percent less capacity than the fixed controller. I treat both conditions as necessary, and check the waiting target separately for steady, periodic, and bursty traffic. This is a sequential decision problem because an extra replica takes time to start. A decision that looks expensive now may prevent a queue later, and a removal that looks efficient now may leave the service unprepared for a burst. That delayed consequence is why reinforcement learning is a reasonable approach. Capacity is measured in replica-seconds, a resource proxy rather than a cloud bill.

### Slide 3: Research that shaped the design

Target: 40 seconds.

Barrett and colleagues and the survey by Gari and colleagues connect reinforcement learning to cloud resource allocation and autoscaling. Wang and colleagues study provisioning in simulation, which supports a simulator as a starting point. PPO provides a clipped policy update, and generalized advantage estimation helps judge which sampled actions were better than expected. Henderson and colleagues emphasize variability across random seeds, which motivated ten training seeds with confidence intervals. AWARE makes clear that production autoscaling needs much more system integration than this project includes. These papers inform the design; the project does not reproduce their results.

### Slide 4: A small, explicit simulator

Target: 50 seconds.

Each episode represents thirty minutes of arrivals. The controller acts every ten seconds, starting with two active replicas. One replica serves twenty requests per second, and a new replica takes twenty seconds to become active. Capacity ranges from one to ten replicas. Requests join a first-in, first-out queue with a thousand-request limit. A request violates the service target if it waits more than two seconds, expires after ten seconds, or is dropped because the queue is full. After arrivals end, I drain the queue so unfinished work still counts. Steady, periodic, and bursty profiles produce seeded Poisson arrivals. These are synthetic workloads, and the waiting metric excludes network and processing time.

### Slide 5: What the agent sees and controls

Target: 40 seconds.

The agent receives nine normalized observations: recent arrivals, a short moving average, the change in demand, queue size and age, active capacity, two stages of starting replicas, and utilization. The actions are remove one replica, hold, or add one. A removal cannot go below one active replica, and an addition cannot exceed ten allocated replicas. Invalid actions become no-ops and are recorded; that detail becomes important later. The observation does not reveal future demand, so the task is partially observable.

### Slide 6: Reward expresses the tradeoff

Target: 45 seconds.

The reward is the negative sum of four penalties. Allocated capacity has a weight of point one, the fraction of resolved requests that violate the target has a weight of five, normalized queue size has a weight of point five, and each successful scaling change costs point zero two. Starting replicas count toward capacity cost. The violation fraction uses requests resolved in the current interval, which is not the same aggregation as the final episode metric. I kept the initial weights rather than tuning them. The large service penalty makes protecting requests attractive, but does not guarantee economical behavior.

### Slide 7: PPO and the actor–critic model

Target: 55 seconds.

The actor assigns probabilities to the three actions, and the critic estimates future discounted return. Both networks have two hidden layers of sixty-four tanh units. During training, the policy samples actions, gathers a rollout, and computes advantages with generalized advantage estimation. The clipped objective limits how far the new action probability can move from the policy that collected the data. The clip range is point two, the discount is point nine nine, and the GAE parameter is point nine five. Adam uses a learning rate of three times ten to the minus four, with an entropy bonus of point zero one. Each rollout has 1,024 decisions, followed by ten epochs in minibatches up to sixty-four. At test time I choose the most probable action, so evaluation is deterministic.

### Slide 8: Implementation and changes to the plan

Target: 45 seconds.

The code separates the simulator, controllers, PPO training, the experiment and ablation runners, and analysis. NumPy handles the networks, manual gradients, and Adam. The biggest change from Part 1 is the framework: PyTorch and Gymnasium were unavailable, so I used NumPy with Gymnasium-style reset and step signatures. DQN remains an alternative, not a completed comparison. Nine tests check the gradients numerically with and without clipping, advantages at terminal steps and at lambda zero and one, the entropy diagnostics, startup timing and request accounting, and seeded reproducibility.

### Slide 9: Training and evaluation protocol

Target: 45 seconds.

I trained ten independent seeds for fifty thousand decisions each. Twenty validation traces select checkpoints and tune the baselines. The test set has thirty different traces, ten per workload type, and every controller receives the same arrivals. Validation selected eight replicas for the fixed controller, and threshold settings of point six and point two with a ten-second cooldown. PPO is evaluated as ten policies on the same thirty traces. That is three hundred policy-trace evaluations, not three hundred independent workloads, so I treat each training seed as one sample and report ninety-five percent confidence intervals across the ten seeds.

### Slide 10: Training improved, but runs differed

Target: 45 seconds.

This chart shows the mean training return across the ten seeds, with dashed lines marking the ninety-five percent confidence interval. Return rises quickly over the first ten thousand decisions and then levels off, and the interval narrows as the seeds converge. Checkpoints were chosen on validation service and cost, not on this curve. Each fifty-thousand-decision run takes only seconds on this machine. Importantly, greedy validation was already favorable before training, so a better training return does not establish that training caused better deterministic service.

### Slide 11: Better average service, higher cost

Target: 60 seconds.

The fixed controller averaged 14,250 replica-seconds and a four point five eight percent violation rate. The threshold controller used 15,905 replica-seconds with six point five nine percent violations. Across ten seeds, PPO averaged about 17,650 replica-seconds and two point two one percent violations, with a confidence interval from about one point five to two point nine percent. Relative to fixed capacity, PPO used twenty-three point eight five percent more resources, the opposite of the ten-percent saving target. A diagnostic is revealing: PPO averaged about 167 invalid actions per episode, out of 180 decisions. The policies keep requesting additions after reaching the capacity limit. Those actions do nothing, so the agent behaves almost like a high fixed-capacity controller.

### Slide 12: Workload and seed results matter

Target: 50 seconds.

Breaking results down by workload changes the interpretation. PPO's mean violation rate is four point one eight percent on steady traffic, two point three eight percent on periodic traffic, and only point zero six percent on bursts. The fixed controller misses eight point three three percent on bursts and the threshold controller thirteen point four seven percent, so extra capacity helps most with bursts. But on steady traffic the PPO confidence interval runs from about two to six point three percent, crossing the five-percent limit. Across seeds, overall violations range from one point eight one to five point one one percent. So the cost condition fails clearly, and the per-workload service condition is not met with confidence.

### Slide 13: Ablations: what clipping and GAE contribute

Target: 55 seconds.

To measure what PPO's components contribute, I retrained ten paired seeds for three variants: no clipping, and two ways of removing GAE, Monte Carlo returns with lambda one and one-step TD with lambda zero. Paired means each seed starts every variant from the same weights and traffic. Without clipping, training reaches the convergence threshold about a thousand decisions sooner, but the largest update KL jumps from about point zero one five to point six, and one run collapsed to minimum capacity by the end. Lambda one is statistically indistinguishable from full PPO here, because episodes are short. Lambda zero is the slowest, about twenty-nine hundred decisions behind, and the least stable. Validation checkpoint selection hides the collapses, so test metrics barely change.

### Slide 14: The policy's entropy hides inactive choices

Target: 45 seconds.

The entropy bonus acts on the policy's three-way action distribution. But the environment turns invalid actions into holds, so I also log the entropy of the actions that actually execute. The two diverge sharply. At the end of training, policy entropy is about point six four nats, but executed entropy is only point one seven, and sixty-three percent of the probability sits on actions the environment rejects. The policy looks exploratory to the loss, but most of that randomness is a choice between add and hold at the ceiling, which both mean hold. That explains the invalid-action counts, and it motivates masking invalid actions.

### Slide 15: Demonstration: actual policy replay

Target: 65 seconds.

I will now play a sixty-second replay of actual evaluated decisions, with three twenty-second segments: steady, periodic, and bursty traffic. Each segment represents thirty simulated minutes. The full curves are visible, and the moving cursor marks the replay time. The gray curve is demand, orange is active capacity, and green is the queue. The bottom panel tracks cumulative violations among resolved requests. Watch the initial capacity ramp and then the large amount of spare capacity. During periodic traffic, demand falls but the policy largely keeps its capacity. On bursty traffic, that spare capacity absorbs peaks. The demo uses seed one, chosen by validation rather than by test appearance, and three predetermined test traces. It illustrates actual behavior, including inefficiency, and does not replace the full test results.

### Slide 16: Reliability and ethical considerations

Target: 45 seconds.

A real autoscaler controls both service access and resource spending. Underprovisioning can delay or drop requests, and the impact may fall unevenly on users; this simulator has no user groups, so it does not evaluate fairness. Overprovisioning wastes money and energy, and replica-seconds are not a measured carbon estimate. A real deployment would need hard capacity and budget limits, a fallback controller, monitoring, and rollback, with exploration kept in a sandbox. I remain accountable for the reward and simulator choices rather than assuming the learned behavior is automatically acceptable.

### Slide 17: What I learned and what comes next

Target: 40 seconds.

The project produced a working simulator, a tested PPO implementation, two baselines, ten-seed results with confidence intervals, component ablations, and a replay. The main lesson is that learning a policy is different from meeting the operational goal. The entropy diagnostic points to the first next step: mask invalid actions during training and evaluation. Then I would revisit reward balance, add stronger baselines, and broaden workloads, with validation for tuning and fresh held-out tests. The defensible conclusion is a useful prototype with clear limitations.

### Slide 18: References and AI disclosure

Target: 25 seconds.

The references connect back to Part 1, and full links are in the repository. OpenAI ChatGPT and Anthropic Claude helped implement and test code, run and analyze the experiments, and draft these slides and narration. I need to review that assistance, verify the sources, and explain the work in my own words. The results come from actual simulator runs.

## Understand these points before presenting

- Replica-seconds are accumulated allocated capacity, including replicas that are starting. They are not dollar costs.
- A violation means a request waits more than two seconds, expires, or overflows the queue. P95 queue waiting only describes served requests and cannot replace the all-request violation metric.
- The ten PPO policies share thirty test traces. Do not describe this as 300 independent workloads. Confidence intervals treat each training seed as one sample and describe seed-to-seed variability on these fixed traces, not uncertainty over all possible workloads.
- A less-negative training return is not evidence that PPO met the service-and-cost objective.
- Most greedy trained policies behave like a high fixed-capacity controller. The untrained validation metrics in the logs can also look favorable under greedy evaluation. These results do not establish that training caused better deterministic service behavior.
- Ablation variants share seeds with full PPO, so their differences are paired. "No GAE" has two forms here: λ = 1 (Monte Carlo) and λ = 0 (one-step TD). Checkpoint selection on validation hides the late-training collapses from the test metrics; the collapses appear in the training curves and logs.
- The policy is categorical, not a squashed Gaussian, so policy entropy is exact. "Executed-action entropy" is the entropy after the environment turns invalid remove/add actions into hold.
- The algorithm is implemented, but DQN, fairness analysis, real traffic, and real deployment are not completed experiments.
- The outcome is a failed operational hypothesis with a useful working prototype. Do not change the numbers to make it look successful.

## Code repository

The project is published at https://github.com/srividyap98/autoscaling_rl . Keep `README.md`, `requirements.txt`, `autoscaling/`, `tests/`, `outputs/`, and `presentation/` in the repository, and check that the instructor has access. Upload the separate slide deck and your own recording as required by the course.

## AI disclosure

OpenAI ChatGPT assisted the original implementation, numerical tests, experiment organization, analysis of measured outputs, and preparation of slides and narration. Anthropic Claude assisted the ten-seed confidence intervals, component ablations, entropy diagnostics, and the corresponding updates to the code, README, slides, and narration. The student must review the code and citations, understand the result interpretation, and create the final recording in their own voice. The recording guide and slide notes are drafts, not a substitute for that review.

## File verification

The deck has 18 slides, editable data charts, and speaker notes that match this guide. Package and schema validation passed, and it was opened and exported to PDF in Microsoft PowerPoint for Mac for a visual check. The demo is H.264, 1280 × 720, exactly 60 seconds, contains no audio, and was regenerated from the validation-selected seed-1 policy of the current run. Nine numerical/simulator tests passed. Reference result timing describes the execution machine and is not a laptop runtime guarantee.
