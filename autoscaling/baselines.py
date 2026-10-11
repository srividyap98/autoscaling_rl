"""Non-learning controls subject to identical environment constraints."""

class Fixed:

    def __init__(self, replicas):
        self.replicas = replicas

    def reset(self):
        pass

    def act(self, obs):
        active = int(round(obs[5] * 10))
        pending = int(round((obs[6] + obs[7]) * 10))
        if active + pending < self.replicas:
            return 2
        if active > self.replicas:
            return 0
        return 1

class Threshold:

    def __init__(self, upper=0.7, lower=0.3, cooldown=20):
        self.upper = upper
        self.lower = lower
        self.cooldown = cooldown
        self.reset()

    def reset(self):
        self.low_count = 0
        self.remaining = 0

    def act(self, obs):
        queue = obs[3]
        util = obs[8]
        self.low_count = self.low_count + 1 if util < self.lower and queue == 0 else 0
        if self.remaining > 0:
            self.remaining -= 10
            return 1
        if util > self.upper or queue > 0:
            self.remaining = self.cooldown - 10
            return 2
        if self.low_count >= 3:
            self.low_count = 0
            self.remaining = self.cooldown - 10
            return 0
        return 1
