"""NumPy clipped PPO, GAE, manual backpropagation and Adam. Separate 9–64–64 MLPs."""
import numpy as np

class MLP:

    def __init__(self, input_size, output_size, rng, output_scale=1):
        self.p = {}
        sizes = [input_size, 64, 64, output_size]
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            self.p[f'w{i}'] = rng.normal(size=(a, b)) * np.sqrt(2 / (a + b)) * (output_scale if i == 2 else 1)
            self.p[f'b{i}'] = np.zeros(b)
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.time = 0

    def forward(self, x):
        h1 = np.tanh(x @ self.p['w0'] + self.p['b0'])
        h2 = np.tanh(h1 @ self.p['w1'] + self.p['b1'])
        return (h2 @ self.p['w2'] + self.p['b2'], (x, h1, h2))

    def backward(self, dy, cache):
        x, h1, h2 = cache
        g = {'w2': h2.T @ dy, 'b2': dy.sum(axis=0)}
        z = dy @ self.p['w2'].T * (1 - h2 * h2)
        g.update(w1=h1.T @ z, b1=z.sum(axis=0))
        z = z @ self.p['w1'].T * (1 - h1 * h1)
        g.update(w0=x.T @ z, b0=z.sum(axis=0))
        return g

    def update(self, g, lr):
        norm = np.sqrt(sum((np.sum(x * x) for x in g.values())))
        scale = min(1, 0.5 / (norm + 1e-12))
        self.time += 1
        for k in self.p:
            grad = g[k] * scale
            self.m[k] = 0.9 * self.m[k] + 0.1 * grad
            self.v[k] = 0.999 * self.v[k] + 0.001 * grad * grad
            self.p[k] -= lr * (self.m[k] / (1 - 0.9 ** self.time)) / (np.sqrt(self.v[k] / (1 - 0.999 ** self.time)) + 1e-08)

def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def actor_loss_gradient(logits, actions, oldlog, advantage, clip=0.2, entropy=0.01):
    p = softmax(logits)
    logp = np.log(np.maximum(p, 1e-20))
    n = len(actions)
    idx = np.arange(n)
    ratio = np.exp(logp[idx, actions] - oldlog)
    clipped = np.clip(ratio, 1 - clip, 1 + clip)
    active = ~((advantage >= 0) & (ratio > 1 + clip) | (advantage < 0) & (ratio < 1 - clip))
    dlog = -advantage * ratio * active / n
    grad = -p * dlog[:, None]
    grad[idx, actions] += dlog
    grad += entropy / n * p * (logp - (p * logp).sum(axis=1, keepdims=True))
    loss = -np.mean(np.minimum(ratio * advantage, clipped * advantage)) + entropy * np.mean((p * logp).sum(axis=1))
    return (loss, grad, dict(entropy=float(-np.mean((p * logp).sum(axis=1))), clip_fraction=float(np.mean(np.abs(ratio - 1) > clip)), approx_kl=float(np.mean(ratio - 1 - (logp[idx, actions] - oldlog)))))

def gae(rewards, values, next_values, terminals, gamma=0.99, lam=0.95):
    """No bootstrapping or recursion through a true terminal boundary."""
    advantage = np.zeros(len(rewards))
    carry = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        mask = 1 - float(terminals[t])
        delta = rewards[t] + gamma * next_values[t] * mask - values[t]
        carry = delta + gamma * lam * mask * carry
        advantage[t] = carry
    return (advantage, advantage + values)

class PPO:

    def __init__(self, seed=0, lr=0.0003, entropy=0.01):
        self.rng = np.random.default_rng(seed)
        self.actor = MLP(9, 3, self.rng, 0.01)
        self.critic = MLP(9, 1, self.rng)
        self.lr = lr
        self.entropy = entropy

    def act(self, obs, deterministic=False):
        z, _ = self.actor.forward(obs[None, :])
        p = softmax(z)[0]
        a = int(np.argmax(p)) if deterministic else int(self.rng.choice(3, p=p))
        v, _ = self.critic.forward(obs[None, :])
        return (a, float(np.log(max(p[a], 1e-20))), float(v[0, 0]))

    def value(self, obs):
        return float(self.critic.forward(obs[None, :])[0][0, 0])

    def update(self, obs, actions, oldlog, advantage, returns, epochs=10, batch=64):
        adv = (advantage - advantage.mean()) / (advantage.std() + 1e-08)
        stats = []
        for _ in range(epochs):
            for ids in np.array_split(self.rng.permutation(len(obs)), int(np.ceil(len(obs) / batch))):
                z, c = self.actor.forward(obs[ids])
                loss, dz, s = actor_loss_gradient(z, actions[ids], oldlog[ids], adv[ids], entropy=self.entropy)
                self.actor.update(self.actor.backward(dz, c), self.lr)
                val, c = self.critic.forward(obs[ids])
                error = val[:, 0] - returns[ids]
                self.critic.update(self.critic.backward((0.5 * error / len(ids))[:, None], c), self.lr)
                s.update(actor_loss=float(loss), value_mse=float(np.mean(error * error)))
                stats.append(s)
        return {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}

    def save(self, path):
        np.savez(path, **{f'{net}_{k}': v for net in ['actor', 'critic'] for k, v in getattr(self, net).p.items()})

    @classmethod
    def load(cls, path):
        m = cls()
        data = np.load(path)
        for net in ['actor', 'critic']:
            for k in getattr(m, net).p:
                getattr(m, net).p[k] = data[f'{net}_{k}']
        return m
