"""Synthetic CSI for the room-classifier tests: a static 64-subcarrier channel
with L-LTF nulls, AGC gain steps and slow drift, where a person moving in a room
modulates that room's own set of multipath components."""
import numpy as np

NULLS = [0, 1] + list(range(27, 38))


class House:
    def __init__(self, labels, seed=0, twins=(), blind=()):
        """twins: pairs of labels given the SAME fingerprint (inseparable);
        blind: rooms whose movement doesn't reach the link at all."""
        self.rng = np.random.default_rng(seed)
        self.base = self.rng.uniform(10, 40, 64)
        self.base[NULLS] = 0
        k = np.arange(64)
        self.print = {}
        for lab in labels:
            centres = self.rng.uniform(2, 62, 3)
            w = sum(np.exp(-0.5 * ((k - c) / self.rng.uniform(2, 5)) ** 2) for c in centres)
            self.print[lab] = dict(w=w / w.max(), shift=self.rng.normal(0, 0.06, 64),
                                   strength=self.rng.uniform(0.15, 0.35))
        for a, b in twins:
            self.print[b] = self.print[a]
        for b in blind:
            self.print[b] = dict(w=np.zeros(64), shift=np.zeros(64), strength=0.0)
        self.print["nobody"] = dict(w=np.zeros(64), shift=np.zeros(64), strength=0.0)

    def record(self, label, secs=30.0, hz=100.0, t0=1.7e9, drift=0.01):
        rng = self.rng
        n = int(secs * hz)
        t = t0 + np.arange(n) / hz + rng.normal(0, 0.5e-3, n)
        f = self.print[label]
        # movement: a few random sway frequencies per subcarrier group
        m = np.zeros((n, 1))
        for _ in range(4):
            m += np.sin(2 * np.pi * rng.uniform(0.3, 2.0) * (t[:, None] - t0) + rng.uniform(0, 6.3))
        m /= 2.0
        prof = self.base * (1 + f["shift"] + rng.normal(0, drift, 64))
        A = prof * (1 + f["strength"] * f["w"] * m * rng.uniform(0.6, 1.4, (n, 64)))
        A = A + rng.normal(0, 0.4, (n, 64))
        gain = np.repeat(rng.uniform(0.85, 1.15, n // 200 + 1), 200)[:n]   # AGC steps
        A = np.clip(A * gain[:, None], 0, 180)
        A[:, NULLS] = 0
        return t, A.astype(np.float32), np.full(n, -45, np.int8)
