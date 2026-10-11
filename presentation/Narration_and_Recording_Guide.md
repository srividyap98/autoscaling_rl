# Narration and recording guide

This is a draft for you to review, understand, and personalize. The target is **14 minutes 5 seconds**, including a narrated 60-second demo. The approximately 2,000-word script requires about 143 words per minute overall. Practice once and adjust to your natural pace; the timing plan is not proof of a 13–15 minute recording.

## Before recording

1. Open `presentation/API_Autoscaling_RL_Presentation.pptx` and `outputs/agent_demo.mp4`. The demo is separate from the deck; preload it in a video player and check the switch from slide 13 to the player and back.
2. Use a screen recorder that captures your slides, webcam, and microphone together. ScreenPal supports this combination: https://screenpal.com/tutorial/record-your-screen-and-webcam . Keep your face visible throughout, including during the demo. Position your webcam overlay where it does not cover chart labels; move it if needed.
3. Keep speaker notes or this guide on a second screen or printout. Record a short sample to check legibility and audio before recording the full presentation.
4. Review the code and sources, explain the outcome in your own words, and retain an accurate AI disclosure. Add your name/course details if required by your instructor.
5. Verify the final recording lasts 13–15 minutes, has your face and voice throughout, includes all three demo segments, and shows slides clearly.

## Timing plan

| Slide | Topic | Time | Cumulative end |
|---|---|---:|---:|
| 1 | Learning to scale an API | 40 s | 0:40 |
| 2 | The goal from Part 1 | 55 s | 1:35 |
| 3 | Research that shaped the design | 45 s | 2:20 |
| 4 | A small, explicit simulator | 60 s | 3:20 |
| 5 | What the agent sees and controls | 50 s | 4:10 |
| 6 | Reward expresses the tradeoff | 55 s | 5:05 |
| 7 | PPO and the actor–critic model | 65 s | 6:10 |
| 8 | Implementation and changes to the plan | 55 s | 7:05 |
| 9 | Training and evaluation protocol | 50 s | 7:55 |
| 10 | Training improved, but runs differed | 50 s | 8:45 |
| 11 | Better average service, higher cost | 70 s | 9:55 |
| 12 | Workload and seed results matter | 55 s | 10:50 |
| 13 | Demonstration: actual policy replay | 65 s | 11:55 |
| 14 | Reliability and ethical considerations | 55 s | 12:50 |
| 15 | What I learned and what comes next | 45 s | 13:35 |
| 16 | References and AI disclosure | 30 s | 14:05 |

## Demo cue sheet

At slide 13, introduce the replay for about five seconds, then switch to the preloaded video. Continue speaking for its full minute. Keep your webcam overlay active. Return to slide 14 after playback.

| Video time | What to discuss |
|---|---|
| 0–20 s | Steady traffic: initial ramp, excess capacity, queue behavior. |
| 20–40 s | Periodic traffic: demand falls while capacity stays high. |
| 40–60 s | Bursts: spare capacity absorbs peaks; this does not prove efficient scaling. |

The full curves remain visible and the cursor indicates replay time. Orange is active capacity/demand panels and violations; gray is observed demand; green is queue. The cumulative violation curve uses resolved requests, while the final episode metric includes all arrivals after queue drainage. The replay uses validation-selected seed 2 and test seeds 20000, 20010, and 20020. It is not a live cloud demo.

## Slide-by-slide draft narration

### Slide 1: Learning to scale an API

Target: 40 seconds.

This project asks whether a reinforcement learning agent can choose how many API replicas to run as demand changes. The challenge is familiar: too little capacity makes requests wait, while too much capacity wastes resources. In Part 1, I proposed a local simulation because I did not have a cloud setup. The completed project follows that approach. I trained PPO, compared it with two simpler controllers, and recorded the agent's decisions. The main finding is that the agent protected service better on average, but did not achieve the planned cost savings.

### Slide 2: The goal from Part 1

Target: 55 seconds.

The original objective had two parts. First, at least 95 percent of requests should meet a two-second waiting target. Second, PPO should use at least ten percent less capacity than the fixed controller. I treat both conditions as necessary for success, and check the waiting target separately for steady, periodic, and bursty traffic. This is a sequential decision problem because an extra replica takes time to start. A decision that looks expensive now may prevent a queue later. A removal that looks efficient now may leave the service unprepared for a burst. That delayed consequence is the reason reinforcement learning is a reasonable approach. Capacity is measured in replica-seconds, which is a resource proxy rather than a cloud bill.

### Slide 3: Research that shaped the design

Target: 45 seconds.

The research helped me choose both the method and the scope. Barrett and colleagues and the survey by Gari and colleagues connect reinforcement learning to cloud resource allocation and autoscaling. Wang and colleagues also study provisioning in simulation, which supports a simulator as a useful starting point. PPO provides a clipped policy update, and generalized advantage estimation helps estimate which sampled actions were better than expected. Henderson and colleagues emphasize variability and reproducibility, which motivated multiple training seeds and separate evaluation workloads. AWARE also makes clear that production autoscaling requires much more system integration than this project includes. These papers inform the design; the project is not a reproduction of their reported production results.

### Slide 4: A small, explicit simulator

Target: 60 seconds.

Each episode represents thirty minutes of arrivals. The controller acts every ten seconds, starting with two active replicas. One replica can serve twenty requests per second, and a new replica takes twenty seconds to become active. The capacity range is one to ten replicas. Requests join a first-in, first-out queue with a thousand-request limit. A request violates the service target if it waits more than two seconds, expires after ten seconds, or is dropped because the queue is full. Once arrivals end, I freeze capacity and drain the queue so unfinished work is still counted. Steady, periodic, and bursty rate profiles produce seeded Poisson arrivals. These are synthetic workloads, not customer traffic. The waiting metric excludes network delay and application processing, so it should not be presented as a complete production latency guarantee.

### Slide 5: What the agent sees and controls

Target: 50 seconds.

The agent receives nine normalized observations. They summarize recent arrivals, a short moving average, the change in demand, queue size and age, active capacity, two stages of replicas still starting, and utilization. The actions are simple: remove one replica, hold capacity, or add one replica. A removal cannot take active capacity below one, and an addition cannot exceed ten allocated replicas. Invalid actions become no-ops and are recorded. The observation is deliberately compact, but it does not reveal future demand or every queue cohort. That makes the task partially observable. The network is feed-forward, so it cannot perfectly reconstruct hidden history. This limitation matters when interpreting why a policy reacts well to some workloads and poorly to others.

### Slide 6: Reward expresses the tradeoff

Target: 55 seconds.

The reward is the negative sum of four penalties. Allocated capacity has a weight of point one, the fraction of resolved requests that violate the target has a weight of five, normalized queue size has a weight of point five, and each successful scaling change costs point zero two. Starting replicas count toward capacity cost before they serve requests. The violation fraction uses requests resolved in the current interval; if none resolve, that term is zero. This is not the same aggregation as the final episode metric, and that mismatch is a limitation. I retained the initial weights rather than tuning them extensively. The much larger service penalty makes protecting requests attractive, but does not guarantee economical behavior. The resulting policy suggests that the balance needs more careful study.

### Slide 7: PPO and the actor–critic model

Target: 65 seconds.

PPO uses an actor to assign probabilities to the three actions and a critic to estimate future discounted return. Both networks have two hidden layers of sixty-four units with tanh activation. Their nine inputs are the observations. The actor has three outputs and the critic has one. During training, the policy samples actions, gathers a rollout, and computes advantages using the critic and generalized advantage estimation. The clipped objective limits how much the new action probability can move relative to the policy that collected the data. Here the clip range is point two, the discount is point nine nine, and the GAE parameter is point nine five. I use Adam with a learning rate of three times ten to the minus four. Each rollout has 1,024 decisions, followed by ten optimization epochs in minibatches up to sixty-four. At test time, I choose the most probable action, so evaluation is deterministic for a fixed workload.

### Slide 8: Implementation and changes to the plan

Target: 55 seconds.

The implementation separates the simulator, controllers, PPO training, experiment runner, and analysis. NumPy handles the networks, manual gradients, and Adam; Matplotlib creates the plots and FFmpeg encodes the replay. The biggest change from Part 1 is the framework. PyTorch and Gymnasium were unavailable in the execution workspace, so the project uses NumPy and Gymnasium-style reset and step signatures. It does not claim to be a registered Gymnasium environment. I kept PPO and the planned network sizes. DQN remains an alternative rather than a completed comparison. Five tests check policy and network gradients numerically, terminal advantage calculations, startup timing and request accounting, and seeded reproducibility. I also corrected a timing boundary so a replica that finishes starting is visible at the next decision. These checks support the implementation, although they cannot prove every modeling assumption is realistic.

### Slide 9: Training and evaluation protocol

Target: 50 seconds.

I trained five independent seeds for fifty thousand decisions each. Twenty validation traces select checkpoints and tune the baseline settings. The final test set has thirty different traces, with ten for each workload type. All controllers receive the same test arrivals and start with two active replicas. Validation selected eight replicas for the fixed controller; it ramps up through the same startup process rather than receiving instant capacity. The threshold controller uses upper utilization point six, lower utilization point two, and a ten-second cooldown. Low utilization must persist for three decisions with an empty queue before downscaling. PPO is evaluated as five trained policies on the same thirty traces. Those are 150 policy-trace evaluations, not 150 independent workloads. Reported summaries average episode metrics. They are descriptive results, without a statistical significance claim.

### Slide 10: Training improved, but runs differed

Target: 50 seconds.

This chart shows mean return over the most recent twenty completed training episodes at five checkpoints. The untrained checkpoint is excluded because it has no completed training episodes. Return generally becomes less negative, but the runs do not follow identical paths. This suggests the sampled training behavior changed, not that the project achieved the practical objective. Checkpoints were chosen using validation service performance and cost, not this curve alone. Four selected policies have test violation rates near one point eight or one point nine percent. Seed two reaches five point one one percent across its test episodes. The total measured time for the five training runs was about forty-eight seconds on this execution machine, excluding plotting and video generation. That confirms the project is computationally small here, but runtime will vary on another computer. Greedy validation was already favorable before training. These results do not establish that training caused better deterministic service performance.

### Slide 11: Better average service, higher cost

Target: 70 seconds.

The fixed controller averaged 14,250 replica-seconds per episode and a four point five eight percent violation rate. The threshold controller used 15,905 replica-seconds with six point five nine percent violations. PPO averaged about 17,600 replica-seconds and two point five zero percent violations. Relative to fixed capacity, PPO used twenty-three point five one percent more resources. That is the opposite of the ten-percent saving target. Its lower violation rate is useful, but it is not enough to call the overall project successful. A diagnostic is especially revealing: PPO averaged about 172 invalid actions per episode, out of 180 decisions. The policies often continue requesting additions after reaching the capacity limit. Those actions do nothing, so the agent can behave almost like a high fixed-capacity controller. The mean test return is also slightly worse than the fixed baseline. This reinforces why separate service and cost metrics are more informative than a favorable training curve.

### Slide 12: Workload and seed results matter

Target: 55 seconds.

Breaking results down changes the interpretation. PPO's mean violation rate is five point zero five percent on steady traffic, two point three nine percent on periodic traffic, and only point zero six percent on bursty traffic. The fixed controller does better on steady traffic, but misses eight point three three percent on bursts. The threshold controller misses thirteen point four seven percent on bursts. Extra capacity is therefore especially helpful in the burst tests, while one weaker PPO run raises the steady average above the five-percent limit. The range across PPO training seeds is about one point eight one to five point one one percent when averaged across all workloads. I would not hide that weaker run or present one favorable demo as the overall outcome. The cost condition fails clearly, and the per-workload service condition also fails narrowly.

### Slide 13: Demonstration: actual policy replay

Target: 65 seconds.

I will now play a sixty-second replay of actual evaluated decisions. It contains three twenty-second segments: steady, periodic, and bursty traffic. Each segment represents thirty simulated minutes. The full curves are visible, and the moving cursor marks the replay time. The gray curve is demand, orange is active service capacity, and green is the queue. The bottom panel tracks cumulative violations among resolved requests. Watch the initial capacity ramp and then the large amount of spare capacity. During periodic traffic, demand falls but the selected policy largely keeps its capacity. On bursty traffic, that spare capacity helps absorb peaks. The demo uses seed two, chosen by validation rather than by test appearance, and three predetermined test traces. It illustrates actual behavior, including inefficiency. It does not demonstrate production readiness or replace the full test results.

### Slide 14: Reliability and ethical considerations

Target: 55 seconds.

A real autoscaler controls both service access and resource spending. Underprovisioning can delay or drop requests, and the impact may fall unevenly on different users or time periods. This simulator has no user identities or priority classes, so it does not evaluate fairness. Overprovisioning can also waste money and energy. Replica-seconds provide a limited resource measure, not a measured carbon estimate. For a real deployment, I would require hard capacity and budget limits, a fallback controller, monitoring, and a way to roll back unsafe behavior. Exploration should happen in a sandbox or carefully controlled setting. These experiments use synthetic data and no live cloud service. I would remain accountable for the choices in the reward and simulator rather than assuming the agent's learned behavior is automatically acceptable.

### Slide 15: What I learned and what comes next

Target: 45 seconds.

The project produced a working local simulator, a trained PPO implementation, two baselines, reproducible logs, and a replay. The main lesson is that learning a policy is different from meeting the intended operational goal. In these tests, the learned controller traded higher capacity for better average service and failed the cost target. Next I would mask invalid actions consistently during both training and evaluation, then investigate reward balance and the difference between interval and episode metrics. I would add stronger predictive or rule-based baselines, more workload variation, and longer repeated experiments. Each change should use validation for tuning and a fresh held-out test set. Real traces would be a later step, after improving the simulator evidence. For now, the defensible conclusion is a useful prototype with clear limitations.

### Slide 16: References and AI disclosure

Target: 30 seconds.

The references here connect back to Part 1, and full links are included in the repository and speaker notes. OpenAI ChatGPT helped implement and test code, organize the experiment, analyze measured outputs, and draft these slides and narration. I need to review and understand that assistance, verify the sources, and explain the final work in my own words. The results shown come from actual simulator runs.

## Understand these points before presenting

- Replica-seconds are accumulated allocated capacity, including replicas that are starting. They are not dollar costs.
- A violation means a request waits more than two seconds, expires, or overflows the queue. P95 queue waiting only describes served requests and cannot replace the all-request violation metric.
- The five PPO policies share thirty test traces. Do not describe this as 150 independent workloads.
- A less-negative training return is not evidence that PPO met the service-and-cost objective.
- Four greedy trained policies mostly behave like a high fixed-capacity controller. The untrained validation metrics in the logs can also look favorable under greedy evaluation. These results do not establish that training caused better deterministic service behavior.
- The algorithm is implemented, but DQN, fairness analysis, real traffic, and real deployment are not completed experiments.
- The outcome is a failed operational hypothesis with a useful working prototype. Do not change the numbers to make it look successful.

## Code repository submission

The ZIP contains runnable code and measured outputs. An accessible remote repository URL is still required. Create an empty GitHub repository, then either upload the extracted project files through GitHub or push this local repository. For a push, use your actual repository URL:

```bash
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

Keep `README.md`, `requirements.txt`, `autoscaling/`, `tests/`, `outputs/`, and `presentation/` in the repository. Check the instructor has access. Upload the separate slide deck and your own recording as required by the course.

## AI disclosure

OpenAI ChatGPT assisted implementation, numerical tests, experiment organization, analysis of measured outputs, and preparation of slides and narration. The student must review the code and citations, understand the result interpretation, and create the final recording in their own voice. The recording guide and slide notes are drafts, not a substitute for that review.

## File verification

The deck has 16 slides, editable data charts, and speaker notes. Package, layout, font, chart/workbook checks and rendered-preview review passed. It has not been opened in native PowerPoint or Google Slides here; open it in your presentation software and check the charts before recording. The demo is H.264, 1280 × 720, exactly 60 seconds, and contains no audio. Five numerical/simulator tests passed. Reference result timing describes the execution machine and is not a laptop runtime guarantee.
