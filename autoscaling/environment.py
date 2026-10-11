"""FIFO simulator. Settings are assumptions, not production measurements."""
from dataclasses import dataclass
from collections import deque
import numpy as np

@dataclass
class Config:
    seconds: int = 1800
    interval: int = 10
    capacity: int = 20
    startup: int = 20
    maximum: int = 10
    queue_limit: int = 1000
    timeout: int = 10
    target: int = 2
    cost_weight: float = 0.1
    violation_weight: float = 5.0
    queue_weight: float = 0.5
    switch_weight: float = 0.02

def traffic(seed, kind, seconds=1800):
    """Exogenous pre-generated traffic permits identical policy comparisons."""
    rng = np.random.default_rng(seed)
    t = np.arange(seconds)
    if kind == 'steady':
        rate = np.full(seconds, rng.uniform(35, 150))
    elif kind == 'periodic':
        low = rng.uniform(20, 40)
        high = rng.uniform(125, 160)
        rate = low + (high - low) * (1 + np.sin(2 * np.pi * t / rng.uniform(300, 600) + rng.uniform(0, 2 * np.pi))) / 2
    elif kind == 'bursty':
        rate = np.full(seconds, rng.uniform(25, 55))
        for start in rng.choice(np.arange(90, seconds - 180, 90), size=5, replace=False):
            duration = int(rng.integers(45, 121))
            rate[start:start + duration] = rng.uniform(130, 180)
    else:
        raise ValueError(kind)
    return (rng.poisson(rate), rate)

class AutoscalingEnv:
    """Gymnasium-style signatures without dependency. Actions remove/hold/add.
    Finite episodes end after arrivals stop AND queue drainage. Terminal GAE
    does not bootstrap. Observation summaries are partially observable.
    """
    obs_size = 9

    def __init__(self, config=None):
        self.c = config or Config()

    def reset(self, seed=0, options=None):
        options = options or {}
        self.kind = options.get('kind', 'periodic')
        self.arrivals, self.rates = traffic(seed, self.kind, self.c.seconds)
        self.t = 0
        self.active = int(options.get('initial', 2))
        self.pending = []
        self.queue = deque()
        self.q = 0
        self.history = deque([0.0] * 3, maxlen=3)
        self.util = 0.0
        self.hist = np.zeros(self.c.timeout + 1, dtype=np.int64)
        self.total_arrivals = self.served = self.failed = self.violations = 0
        self.cost = self.switches = self.invalid = self.max_queue = 0
        self.done = False
        return (self.observe(), {})

    def observe(self):
        age = self.t - self.queue[0][0] if self.queue else 0
        return np.array([self.history[-1] / 200, np.mean(self.history) / 200, (self.history[-1] - self.history[-2]) / 200, self.q / 1000, age / 10, self.active / 10, sum((0 < x - self.t <= 10 for x in self.pending)) / 10, sum((x - self.t > 10 for x in self.pending)) / 10, self.util], dtype=np.float64)

    def _tick(self, arrivals):
        self.active += sum((x <= self.t for x in self.pending))
        self.pending = [x for x in self.pending if x > self.t]
        expired = 0
        while self.queue and self.t - self.queue[0][0] > self.c.timeout:
            _, n = self.queue.popleft()
            self.q -= n
            expired += n
        accepted = min(arrivals, self.c.queue_limit - self.q)
        overflow = arrivals - accepted
        if accepted:
            self.queue.append([self.t, accepted])
            self.q += accepted
        self.total_arrivals += arrivals
        self.failed += expired + overflow
        self.violations += expired + overflow
        cap = self.active * self.c.capacity
        remaining = cap
        served = bad = 0
        while self.queue and remaining:
            born, count = self.queue[0]
            n = min(count, remaining)
            wait = self.t - born
            self.hist[wait] += n
            bad += n if wait > self.c.target else 0
            served += n
            remaining -= n
            self.q -= n
            if n == count:
                self.queue.popleft()
            else:
                self.queue[0][1] -= n
        self.served += served
        self.violations += bad
        self.cost += self.active + len(self.pending)
        self.max_queue = max(self.max_queue, self.q)
        self.t += 1
        self.active += sum((x <= self.t for x in self.pending))
        self.pending = [x for x in self.pending if x > self.t]
        return (served + expired + overflow, bad + expired + overflow, served / cap, self.q)

    def valid_actions(self):
        """Remove/hold/add mask; an invalid action is executed as hold."""
        return np.array([self.active > 1, True, self.active + len(self.pending) < self.c.maximum])

    def step(self, action):
        if self.done:
            raise RuntimeError('reset required')
        if action not in (0, 1, 2):
            raise ValueError(action)
        changed = 0
        if action == 2:
            if self.active + len(self.pending) < self.c.maximum:
                self.pending.append(self.t + self.c.startup)
                changed = 1
            else:
                self.invalid += 1
        elif action == 0:
            if self.active > 1:
                self.active -= 1
                changed = 1
            else:
                self.invalid += 1
        self.switches += changed
        resolved = bad = qs = us = 0
        begin = self.t
        oldcost = self.cost
        for _ in range(self.c.interval):
            if self.t >= self.c.seconds:
                break
            r, b, u, q = self._tick(int(self.arrivals[self.t]))
            resolved += r
            bad += b
            qs += q
            us += u
        n = self.t - begin
        self.history.append(float(np.sum(self.arrivals[begin:self.t]) / max(n, 1)))
        self.util = us / max(n, 1)
        self.done = self.t >= self.c.seconds
        if self.done:
            while self.q:
                r, b, _, q = self._tick(0)
                resolved += r
                bad += b
                qs += q
        seconds = self.t - begin
        reward = -(self.c.cost_weight * (self.cost - oldcost) / max(seconds, 1) / 10 + self.c.violation_weight * bad / max(resolved, 1) + self.c.queue_weight * qs / max(seconds, 1) / 1000 + self.c.switch_weight * changed)
        info = dict(time=self.t, rate=self.history[-1], active=self.active, pending=len(self.pending), queue=self.q, util=self.util, interval_bad=bad, interval_resolved=resolved, changed=changed)
        if self.done:
            info['metrics'] = self.metrics()
        return (self.observe(), float(reward), self.done, False, info)

    def metrics(self):
        p95 = int(np.searchsorted(np.cumsum(self.hist), 0.95 * self.served)) if self.served else None
        assert self.total_arrivals == self.served + self.failed + self.q
        return dict(arrivals=self.total_arrivals, served=self.served, failed=self.failed, violation_rate=self.violations / max(self.total_arrivals, 1), replica_seconds=self.cost, p95_wait_seconds=p95, scaling_actions=self.switches, invalid_actions=self.invalid, max_queue=self.max_queue)
