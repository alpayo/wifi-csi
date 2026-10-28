"""
SGP CSI WIFI — room classifier: "which room is someone moving in?"

One card and one receiver cannot LOCATE a person: a single link has no geometry
to triangulate with. What it can do is recognise a learned fingerprint. A body
moving in a given room perturbs a particular set of multipath components, and
that shows up as a characteristic pattern across the CSI subcarriers. So rooms
are taught, not computed: you record a short walk-around in every room (guided
from the House map page, web/rooms.html), and this module learns the patterns
and MEASURES how well they separate on your house, your card and your link.

Two modes, chosen automatically (like engine.py):
  • KNN     — numpy only, always available once the rooms are recorded. A
              local-mean k-nearest-neighbour classifier over a per-window profile
              of every subcarrier's gain-normalised median and log-spread.
  • NEURAL  — if torch is installed AND training produced a room CSINet that beat
              the numpy model on the same held-out recordings (rooms_neural.py).
  Untrained, the engine says so (mode "untrained") and never names a room.

Honesty rules the rest of the design:
  • Accuracy is cross-validated by holding out a whole PASS (every room is
    recorded in two separate passes through the house), so a score can't come
    from the classifier recognising the recording session instead of the room.
  • Rooms whose cross-validated pairwise separation is near chance are merged
    into one zone; the page then lights the zone, never a single room in it.
  • "Nobody moving" is a recorded class, not a threshold, so an empty house
    reads as empty instead of being forced into the nearest room.
"""
import importlib, json, os, threading, time
from collections import deque
import numpy as np

NOBODY = "nobody"
# The rooms traced on web/floorplan.png (ids must match the page's polygons).
# The unnamed hallway belongs to the Entrance zone and the balcony behind the
# glass partition to the Kitchen zone, so every walkable space has a zone.
ROOMS = [
    ("bedroom",  "Bedroom"),
    ("bath2",    "Bath 2"),
    ("bath1",    "Bath 1"),
    ("cem",      "Cem"),
    ("deniz",    "Deniz"),
    ("ece",      "Ece"),
    ("kitchen",  "Kitchen"),
    ("entrance", "Entrance"),
    ("bath3",    "Bath 3"),
    ("living",   "Living Room"),
]
LABELS = [NOBODY] + [r for r, _ in ROOMS]
NAMES = dict([(NOBODY, "Nobody moving")] + ROOMS)

N_SUB = 64            # L-LTF subcarriers per frame (shorter frames are zero-padded)
MAX_HZ = 100          # frames kept per second (extra frames are dropped, not averaged)
WIN_S = 2.0           # seconds of CSI behind every decision
HOP_S = 0.5           # training-window hop
MIN_FRAMES = 20       # a window needs at least this many frames
INFER_EVERY_S = 0.25  # live decision cadence
STALE_S = 1.5         # no frame for this long -> no signal, nothing is shown
K = 7                 # neighbours per class in the local-mean k-NN
SMOOTH = 0.72         # temporal smoothing, same constant as AIEngine
PAIR_MIN = 0.75       # pairwise CV separation below this -> rooms are merged
RELIABLE_MIN = 0.60   # zone CV recall below this -> shown as low confidence
N_BLOCKS = 5          # time blocks for a class that has only one recording

KNN_FILE = "rooms_knn.npz"
REPORT_FILE = "rooms_report.json"
NEURAL_FILE = "rooms_neural.pt"
NEURAL_META = "rooms_neural.json"


# ──────────────────────────────────────────────────────────────────────────
# Live frame buffer
# ──────────────────────────────────────────────────────────────────────────
class FrameBuffer:
    """Recent CSI amplitude frames for one card, at most MAX_HZ per second.

    Frames beyond MAX_HZ are DROPPED rather than averaged, so per-frame noise —
    and therefore the features — do not depend on how fast the card is pinged.
    Fed from the UDP thread, read from the DSP thread, hence the lock.
    """
    def __init__(self, keep_s=8.0, max_hz=MAX_HZ):
        n = int(keep_s * max_hz) + 1
        self._t = deque(maxlen=n)
        self._a = deque(maxlen=n)
        self._r = deque(maxlen=n)
        self._max_hz = max_hz
        self._slot = None
        self._lock = threading.Lock()
        self.seq = 0                  # frames accepted so far

    def push(self, t, amp, rssi=0):
        slot = int(t * self._max_hz)
        if slot == self._slot:
            return
        a = np.zeros(N_SUB, np.float32)
        n = min(N_SUB, len(amp))
        a[:n] = amp[:n]
        with self._lock:
            self._slot = slot
            self._t.append(float(t)); self._a.append(a); self._r.append(int(rssi))
            self.seq += 1

    def window(self, secs):
        """(t, A) for the frames of the last `secs` seconds (A is T x N_SUB)."""
        with self._lock:
            if not self._t:
                return np.zeros(0), np.zeros((0, N_SUB), np.float32)
            t = np.array(self._t)
            i = int(np.searchsorted(t, t[-1] - secs))
            A = np.array(list(self._a)[i:], np.float32)
        return t[i:], A

    def since(self, seq):
        """Frames accepted after `seq`: (new_seq, t, A, rssi)."""
        with self._lock:
            n = min(self.seq - seq, len(self._t))
            if n <= 0:
                return self.seq, np.zeros(0), np.zeros((0, N_SUB), np.float32), np.zeros(0, np.int8)
            t = np.array(list(self._t)[-n:])
            A = np.array(list(self._a)[-n:], np.float32)
            r = np.array(list(self._r)[-n:], np.int8)
            return self.seq, t, A, r

    def last_time(self):
        with self._lock:
            return self._t[-1] if self._t else None


# ──────────────────────────────────────────────────────────────────────────
# Features
# ──────────────────────────────────────────────────────────────────────────
def select_active(A):
    """Subcarriers that carry signal in (nearly) every frame: drops the DC and
    guard-band nulls and any carrier the card sometimes zeroes."""
    energy = A.mean(axis=0)
    zero_frac = (A < 0.5).mean(axis=0)
    act = np.where((energy > 0.15 * energy.max()) & (zero_frac < 0.02))[0]
    if len(act) < 4:
        raise ValueError("too few usable subcarriers — is the card streaming CSI?")
    return act


def gain_normalise(A, active):
    """Divide every frame by its mean amplitude, removing the radio's AGC gain
    steps; frames with no signal are dropped."""
    X = A[:, active].astype(np.float64)
    g = X.mean(axis=1)
    ok = g > 0.5
    return X[ok] / g[ok, None]


def window_features(A, active):
    """Feature vector for one window of raw amplitudes A (T x N_SUB): every
    active subcarrier's median (the static multipath shape, which a body in the
    room changes) and log robust spread (how strongly movement modulates it)."""
    X = gain_normalise(A, active)
    if len(X) < MIN_FRAMES:
        return None
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0) * 1.4826
    return np.concatenate([med, np.log(mad + 1e-3)])


def window_starts(t):
    """Start times of the WIN_S training windows (HOP_S apart) across t."""
    if len(t) == 0 or t[-1] - t[0] < WIN_S:
        return np.zeros(0)
    return np.arange(t[0], t[-1] - WIN_S + 1e-9, HOP_S)


# ──────────────────────────────────────────────────────────────────────────
# Local-mean k-nearest-neighbour classifier
# ──────────────────────────────────────────────────────────────────────────
class LocalMeanKNN:
    """For each class, the mean distance from a sample to its k nearest training
    windows of that class. Unlike a nearest centroid this copes with a room
    whose fingerprint has several modes (a long hallway, a room with a door).
    Features are z-scored on the training data only."""
    def __init__(self, n_cls, k=K):
        self.n_cls, self.k = n_cls, k

    def fit(self, X, y):
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-6
        self.Z = ((X - self.mu) / self.sd).astype(np.float32)
        self.y = np.asarray(y)
        return self

    def distances(self, X):
        """(N, n_cls) local-mean distances, scaled by sqrt(#features);
        inf for a class with no training windows."""
        Z = ((np.atleast_2d(X) - self.mu) / self.sd).astype(np.float32)
        d2 = (Z * Z).sum(1)[:, None] + (self.Z * self.Z).sum(1)[None, :] - 2.0 * Z @ self.Z.T
        D = np.sqrt(np.maximum(d2, 0.0)) / np.sqrt(Z.shape[1])
        out = np.full((len(Z), self.n_cls), np.inf)
        for c in range(self.n_cls):
            Dc = D[:, self.y == c]
            if Dc.shape[1] == 0:
                continue
            kk = min(self.k, Dc.shape[1])
            out[:, c] = np.partition(Dc, kk - 1, axis=1)[:, :kk].mean(axis=1)
        return out


def softmax_from_distances(D, temp):
    logits = -np.where(np.isfinite(D), D, 1e6) / temp
    logits -= logits.max(axis=1, keepdims=True)
    P = np.exp(logits)
    return P / P.sum(axis=1, keepdims=True)


def fit_temperature(score, y, to_probs):
    """Temperature that makes held-out probabilities honest (minimum negative
    log-likelihood of the true class). This is what makes the confidence the
    page shows a calibrated one."""
    best_t, best_nll = 1.0, np.inf
    for temp in np.logspace(-3, 1.5, 91):
        P = to_probs(score, temp)
        nll = -np.mean(np.log(P[np.arange(len(y)), y] + 1e-9))
        if nll < best_nll:
            best_t, best_nll = float(temp), float(nll)
    return best_t, best_nll


def hedge_eps(n_recordings):
    """Uniform probability mixed into every output. Windows of one recording are
    not independent evidence — roughly one trial per held-out recording — so the
    confidence shown never claims more certainty than that many trials support."""
    return 1.0 / (n_recordings + 2)


def hedge(P, eps):
    return (1.0 - eps) * P + eps / P.shape[-1]


# ──────────────────────────────────────────────────────────────────────────
# Recordings -> dataset
# ──────────────────────────────────────────────────────────────────────────
def recordings_dir(data_dir):
    return os.path.join(data_dir, "recordings")


def save_recording(data_dir, label, pass_no, node, t, A, rssi):
    """Write one room recording atomically; it replaces any older recording of
    the same room and pass. Returns the file name."""
    d = recordings_dir(data_dir)
    os.makedirs(d, exist_ok=True)
    fn = "%s_p%d_%d.npz" % (label, pass_no, int(t[0] * 1000))
    tmp = os.path.join(d, fn + ".part")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, label=label, pass_no=pass_no, node=node, started=float(t[0]),
                            t=np.asarray(t, np.float64), amp=np.asarray(A).astype(np.float16),
                            rssi=np.asarray(rssi).astype(np.int8))
    os.replace(tmp, os.path.join(d, fn))
    prefix = "%s_p%d_" % (label, pass_no)
    for old in os.listdir(d):
        if old.startswith(prefix) and old.endswith(".npz") and old != fn:
            os.remove(os.path.join(d, old))
    return fn


def load_recordings(data_dir):
    """Every saved recording: dicts with label, pass, node, t, A, file. Only the
    newest recording of each room and pass counts."""
    out = {}
    d = recordings_dir(data_dir)
    if not os.path.isdir(d):
        return out
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".npz"):
            continue
        try:
            z = np.load(os.path.join(d, fn), allow_pickle=False)
            lab = str(z["label"])
            if lab not in LABELS:
                continue
            key, started = (lab, int(z["pass_no"])), float(z["started"])
            if key in out and out[key]["started"] >= started:
                continue
            out[key] = dict(label=lab, pass_no=key[1], node=str(z["node"]),
                            started=started, t=z["t"].astype(np.float64),
                            A=z["amp"].astype(np.float32), file=fn)
        except Exception:
            continue                       # a half-written or foreign file
    return sorted(out.values(), key=lambda r: r["file"])


def build_dataset(recs, active=None, raw_steps=None):
    """Windows of every recording -> X (features), y, rec (recording index),
    pas (its pass), tmid (window centre, seconds into its recording) and, with raw_steps, the
    gain-normalised raw windows resampled to raw_steps frames (for the CNN)."""
    if active is None:
        active = select_active(np.concatenate([r["A"] for r in recs]))
    X, y, rec, pas, tmid, raw = [], [], [], [], [], []
    for ri, r in enumerate(recs):
        t, A = r["t"], r["A"]
        for s in window_starts(t):
            i0, i1 = np.searchsorted(t, [s, s + WIN_S])
            f = window_features(A[i0:i1], active)
            if f is None:
                continue
            X.append(f); y.append(LABELS.index(r["label"])); rec.append(ri)
            pas.append(r["pass_no"]); tmid.append(s + WIN_S / 2 - t[0])
            if raw_steps:
                G = gain_normalise(A[i0:i1], active)
                raw.append(G[np.linspace(0, len(G) - 1, raw_steps).round().astype(int)])
    ds = dict(X=np.array(X), y=np.array(y, int), rec=np.array(rec, int),
              pas=np.array(pas, int), tmid=np.array(tmid), active=active)
    if raw_steps:
        ds["raw"] = np.array(raw, np.float32)
    return ds


# ──────────────────────────────────────────────────────────────────────────
# Cross-validation, separability and zones
# ──────────────────────────────────────────────────────────────────────────
def two_pass_classes(y, pas):
    return [c for c in np.unique(y) if len(np.unique(pas[y == c])) >= 2]


def cv_folds(y, rec, pas, tmid):
    """(test, train) masks. Rooms recorded in both passes are tested one whole
    pass at a time, by a model trained on the other pass: a recording session it
    never saw, and one recording of EVERY class to learn from. (Holding out
    single recordings instead would leave the tested room with half the training
    data of the others, which drags the score below chance when rooms barely
    differ.) A room with one recording falls back to holding out one of N_BLOCKS
    time blocks, minus every training window that overlaps it — still
    optimistic, and the report says so for that room."""
    folds = []
    two = np.isin(y, two_pass_classes(y, pas))
    for p in np.unique(pas[two]):
        test = two & (pas == p)
        folds.append((test, ~test))
    for r in np.unique(rec[~two]):
        m = rec == r
        edges = np.linspace(tmid[m].min(), tmid[m].max() + 1e-9, N_BLOCKS + 1)
        for b in range(N_BLOCKS):
            test = m & (tmid >= edges[b]) & (tmid < edges[b + 1])
            if not test.any():
                continue
            lo, hi = tmid[test].min(), tmid[test].max()
            overlap = m & (tmid > lo - WIN_S) & (tmid < hi + WIN_S)
            folds.append((test, ~overlap))
    return folds


def pairwise_separation(P, y, n_cls):
    """Balanced accuracy of telling each pair of classes apart from held-out
    probabilities (0.5 = chance, 1 = always right)."""
    S = np.ones((n_cls, n_cls))
    for i in range(n_cls):
        for j in range(i + 1, n_cls):
            mi, mj = y == i, y == j
            if not mi.any() or not mj.any():
                continue
            ri = np.mean(P[mi, i] >= P[mi, j])
            rj = np.mean(P[mj, j] > P[mj, i])
            S[i, j] = S[j, i] = 0.5 * (ri + rj)
    return S


def merge_zones(S, n_cls, thr=PAIR_MIN):
    """Union of every pair the data can't separate -> zones (lists of class
    indices). Transitive on purpose: A~B and B~C means one zone A+B+C."""
    parent = list(range(n_cls))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for i in range(n_cls):
        for j in range(i + 1, n_cls):
            if S[i, j] < thr:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n_cls):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: g[0])


def evaluate(P, y, n_cls, scheme):
    """Report on held-out probabilities P: accuracy, per-class recall, the
    confusion matrix, the zones the data supports and each zone's accuracy."""
    pred = P.argmax(axis=1)
    conf = np.zeros((n_cls, n_cls), int)
    np.add.at(conf, (y, pred), 1)
    rows = conf.sum(axis=1)
    cols = conf.sum(axis=0)
    recall = np.where(rows > 0, np.diag(conf) / np.maximum(rows, 1), np.nan)
    precision = np.where(cols > 0, np.diag(conf) / np.maximum(cols, 1), np.nan)
    S = pairwise_separation(P, y, n_cls)
    zones = merge_zones(S, n_cls)
    zone_of = np.zeros(n_cls, int)
    for zi, g in enumerate(zones):
        zone_of[g] = zi
    zone_rows = []
    for zi, g in enumerate(zones):
        m = np.isin(y, g)
        acc = float(np.mean(zone_of[pred[m]] == zi)) if m.any() else None
        zone_rows.append(dict(members=[LABELS[i] for i in g], recall=acc,
                              reliable=bool(len(g) == 1 and acc is not None and acc >= RELIABLE_MIN)))
    confusable = [dict(a=LABELS[i], b=LABELS[j], sep=round(float(S[i, j]), 3))
                  for i in range(n_cls) for j in range(i + 1, n_cls) if S[i, j] < PAIR_MIN]
    return dict(
        scheme=scheme, n=int(len(y)),
        accuracy=float(np.mean(pred == y)),
        balancedAccuracy=float(np.nanmean(recall)),
        chance=1.0 / n_cls,
        zoneAccuracy=float(np.mean(zone_of[pred] == zone_of[y])),
        perClass=[dict(label=LABELS[c], name=NAMES[LABELS[c]], n=int(rows[c]),
                       recall=None if np.isnan(recall[c]) else round(float(recall[c]), 3),
                       precision=None if np.isnan(precision[c]) else round(float(precision[c]), 3))
                  for c in range(n_cls)],
        confusion=conf.tolist(),
        separation=np.round(S, 3).tolist(),
        confusable=confusable,
        zones=zone_rows,
    )


# ──────────────────────────────────────────────────────────────────────────
# Training (called from the page via room_calib.py, or train_rooms.py)
# ──────────────────────────────────────────────────────────────────────────
def _torch_available():
    try:
        importlib.import_module("torch")     # a real import: a broken install counts as absent
        return True
    except Exception:
        return False


def train(data_dir, progress=lambda msg: None, neural=None):
    """Train on every recording in data_dir; write the model + report there.
    Returns the report of the model that will run. Raises ValueError with a
    human-readable reason when the recordings aren't enough to train on."""
    progress("loading recordings")
    recs = load_recordings(data_dir)
    have = {r["label"] for r in recs}
    missing = [NAMES[l] for l in LABELS if l not in have]
    if missing:
        raise ValueError("record these first: " + ", ".join(missing))
    nodes = sorted({r["node"] for r in recs})
    if len(nodes) > 1:
        raise ValueError("recordings come from different cards (%s) — discard and re-record"
                         % ", ".join(nodes))
    if neural is None:
        neural = _torch_available()
    neural_mod = None
    if neural:
        from . import rooms_neural
        neural_mod = rooms_neural
    ds = build_dataset(recs, raw_steps=neural_mod.T_STEPS if neural_mod else None)
    X, y, rec, pas, tmid = ds["X"], ds["y"], ds["rec"], ds["pas"], ds["tmid"]
    n_cls = len(LABELS)

    # --- numpy k-NN: out-of-fold distances for every window ---
    progress("cross-validating (%d windows, %d recordings)" % (len(y), len(recs)))
    D = np.full((len(y), n_cls), np.inf)
    for test, tr in cv_folds(y, rec, pas, tmid):
        D[test] = LocalMeanKNN(n_cls).fit(X[tr], y[tr]).distances(X[test])
    temp, nll = fit_temperature(D, y, softmax_from_distances)
    P = softmax_from_distances(D, temp)
    two = two_pass_classes(y, pas)
    per_class_scheme = {LABELS[c]: "other-pass" if c in two else "within-recording"
                        for c in range(n_cls)}
    report = evaluate(P, y, n_cls, "held-out passes (trained on one pass, tested on the other)"
                      if len(two) == n_cls else
                      "held-out passes where a room has both, held-out time blocks of its "
                      "one recording where it doesn't (optimistic)")
    for row in report["perClass"]:
        row["scheme"] = per_class_scheme[row["label"]]
    eps = hedge_eps(len(recs))
    report.update(mode="knn", temperature=temp, nll=round(nll, 4), hedge=eps, node=nodes[0],
                  recordings=len(recs), trainedAt=time.time(),
                  labels=LABELS, names=[NAMES[l] for l in LABELS])
    model = LocalMeanKNN(n_cls).fit(X, y)
    np.savez_compressed(os.path.join(data_dir, KNN_FILE),
                        active=ds["active"], mu=model.mu, sd=model.sd, Z=model.Z,
                        y=model.y.astype(np.int16), k=model.k, temperature=temp, hedge=eps,
                        labels=np.array(LABELS), node=nodes[0])
    # an older neural model must never outlive a retrain on different data
    for stale in (NEURAL_FILE, NEURAL_META):
        if os.path.exists(os.path.join(data_dir, stale)):
            os.remove(os.path.join(data_dir, stale))

    _write_json(os.path.join(data_dir, REPORT_FILE), report)   # the k-NN stands on its own

    # --- optional upgrade: room CSINet, kept only if it beats k-NN held-out ---
    if neural_mod is not None:
        try:
            nrep, trial = neural_mod.train_and_compare(ds, data_dir, report, progress)
        except Exception as e:                      # the k-NN stays in charge
            nrep, trial = None, dict(kept=False, error=repr(e))
        report["neuralTrial"] = trial
        _write_json(os.path.join(data_dir, REPORT_FILE), report)
        if nrep is not None:
            return nrep
    return report


def _write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


# ──────────────────────────────────────────────────────────────────────────
# Live inference
# ──────────────────────────────────────────────────────────────────────────
class KNNModel:
    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        if [str(l) for l in z["labels"]] != LABELS:
            raise ValueError("room list changed since training — retrain")
        self.active = z["active"]
        self.node = str(z["node"])
        self.temperature = float(z["temperature"])
        self.hedge = float(z["hedge"])
        self.knn = LocalMeanKNN(len(LABELS), int(z["k"]))
        self.knn.mu, self.knn.sd, self.knn.Z, self.knn.y = z["mu"], z["sd"], z["Z"], z["y"].astype(int)

    def proba(self, A):
        f = window_features(A, self.active)
        if f is None:
            return None
        return hedge(softmax_from_distances(self.knn.distances(f), self.temperature)[0], self.hedge)


class RoomEngine:
    """Runs the trained room model on a card's live FrameBuffer and smooths the
    result over time. Every output says which mode produced it."""
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self._lock = threading.Lock()      # reload() runs on the training thread
        self.reload()

    def reload(self):
        with self._lock:
            self._load()

    def _load(self):
        self.model, self.report, self.mode, self.error = None, None, "untrained", None
        try:
            nm = os.path.join(self.data_dir, NEURAL_META)
            if os.path.exists(os.path.join(self.data_dir, NEURAL_FILE)) and _torch_available():
                from . import rooms_neural
                self.model = rooms_neural.NeuralModel(self.data_dir)
                self.report = _read_json(nm)
                self.mode = "neural"
        except Exception as e:
            self.model, self.error = None, "neural model not loaded: %s" % e
        if self.model is None and os.path.exists(os.path.join(self.data_dir, KNN_FILE)):
            try:
                self.model = KNNModel(os.path.join(self.data_dir, KNN_FILE))
                self.report = _read_json(os.path.join(self.data_dir, REPORT_FILE))
                self.mode = "knn"
            except Exception as e:
                self.model, self.report, self.error = None, None, "room model not loaded: %s" % e
        self.zones = [z["members"] for z in self.report["zones"]] if self.report else []
        self.zone_ok = [z["reliable"] for z in self.report["zones"]] if self.report else []
        self.smooth = None
        self._next = 0.0
        self._out = None

    def infer(self, buf, card, now=None):
        with self._lock:
            return self._infer(buf, card, time.time() if now is None else now)

    def _infer(self, buf, card, now):
        base = dict(mode=self.mode, labels=LABELS, names=[NAMES[l] for l in LABELS],
                    error=self.error, node=getattr(self.model, "node", None))
        if self.model is None:
            return dict(base, state="untrained")
        if card != self.model.node:
            self.smooth = None
            return dict(base, state="other-card")
        last = buf.last_time()
        if last is None or now - last > STALE_S:
            self.smooth = None
            return dict(base, state="no-signal")
        if now >= self._next or self._out is None:
            self._next = now + INFER_EVERY_S
            _, A = buf.window(WIN_S)
            p = self.model.proba(A) if len(A) >= MIN_FRAMES else None
            if p is None:
                self.smooth = None
                return dict(base, state="no-signal")
            self.smooth = p if self.smooth is None else SMOOTH * self.smooth + (1 - SMOOTH) * p
            self._out = self._summarise(self.smooth)
        return dict(base, state="live", **self._out)

    def _summarise(self, p):
        zp = [float(sum(p[LABELS.index(l)] for l in z)) for z in self.zones]
        zi = int(np.argmax(zp))
        return dict(probs=[round(float(x), 3) for x in p],
                    zoneProbs=[round(x, 3) for x in zp],
                    zone=zi, top=self.zones[zi], conf=round(zp[zi], 3),
                    merged=len(self.zones[zi]) > 1, reliable=bool(self.zone_ok[zi]))


def _read_json(path):
    with open(path) as f:
        return json.load(f)
