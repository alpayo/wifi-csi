"""
SGP CSI WIFI — optional neural room classifier (needs torch).

The same CSINet as the activity model (model.py), with one output per room
class, trained on the room recordings. rooms.train() calls this only when torch
is installed, and the result is KEPT only if it beats the numpy k-NN on the same
held-out recordings — otherwise the k-NN stays in charge and the report says
what the neural model scored. Once kept, RoomEngine switches to it by itself.
"""
import json, os
import numpy as np
import torch
import torch.nn as nn
from .model import CSINet
from . import rooms

T_STEPS = 48         # frames per window fed to the network (resampled from WIN_S)
EPOCHS = 30


def _prep(W):
    """Per-window normalisation, as train.py and AIEngine do."""
    return (W - W.mean(axis=(1, 2), keepdims=True)) / (W.std(axis=(1, 2), keepdims=True) + 1e-6)


def _fit(W, y, n_cls, progress, tag, seed=0):
    torch.manual_seed(seed)
    m = CSINet(n_cls=n_cls)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    counts = np.bincount(y, minlength=n_cls).astype(np.float32)
    w = torch.tensor(counts.sum() / np.maximum(counts, 1) / n_cls)
    lossf = nn.CrossEntropyLoss(weight=w)
    Xt, Yt = torch.tensor(_prep(W)), torch.tensor(y, dtype=torch.long)
    for ep in range(EPOCHS):
        m.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), 32):
            b = perm[i:i + 32]
            opt.zero_grad(); lossf(m(Xt[b]), Yt[b]).backward(); opt.step()
        progress("neural %s: epoch %d/%d" % (tag, ep + 1, EPOCHS))
    m.eval()
    return m


def _logits(m, W):
    with torch.no_grad():
        return m(torch.tensor(_prep(W))).numpy()


def softmax_from_logits(L, temp):
    z = L / temp
    z = z - z.max(axis=1, keepdims=True)
    P = np.exp(z)
    return P / P.sum(axis=1, keepdims=True)


def holdout_split(ds):
    """Test = the last pass of every room recorded in both passes, or the last
    quarter (minus overlapping windows) of a room's only recording."""
    y, rec, pas, tmid = ds["y"], ds["rec"], ds["pas"], ds["tmid"]
    two = np.isin(y, rooms.two_pass_classes(y, pas))
    test = two & (pas == (pas[two].max() if two.any() else 0))
    train = ~test
    for r in np.unique(rec[~two]):
        m = rec == r
        cut = np.quantile(tmid[m], 0.75)
        test |= m & (tmid >= cut)
        train &= ~(m & (tmid > cut - rooms.WIN_S))
    return test, train


def train_and_compare(ds, data_dir, knn_report, progress):
    """Returns (neural report or None, trial summary for the k-NN report)."""
    W, y = ds["raw"], ds["y"]
    n_cls = len(rooms.LABELS)
    test, train = holdout_split(ds)
    knn = rooms.LocalMeanKNN(n_cls).fit(ds["X"][train], y[train])
    knn_acc = float(np.mean(knn.distances(ds["X"][test]).argmin(axis=1) == y[test]))
    m = _fit(W[train], y[train], n_cls, progress, "held-out check")
    L = _logits(m, W[test])
    nn_acc = float(np.mean(L.argmax(axis=1) == y[test]))
    trial = dict(neuralAcc=round(nn_acc, 3), knnAcc=round(knn_acc, 3), n=int(test.sum()),
                 kept=bool(nn_acc > knn_acc))
    if not trial["kept"]:
        return None, trial
    temp, nll = rooms.fit_temperature(L, y[test], softmax_from_logits)
    report = rooms.evaluate(softmax_from_logits(L, temp), y[test], n_cls,
                            "held-out last pass (one split)")
    eps = rooms.hedge_eps(len(np.unique(ds["rec"][test])))
    report.update(mode="neural", temperature=temp, nll=round(nll, 4), hedge=eps, node=knn_report["node"],
                  recordings=knn_report["recordings"], trainedAt=knn_report["trainedAt"],
                  labels=knn_report["labels"], names=knn_report["names"], neuralTrial=trial)
    final = _fit(W, y, n_cls, progress, "final")
    torch.save(final.state_dict(), os.path.join(data_dir, rooms.NEURAL_FILE))
    meta = dict(report, active=[int(a) for a in ds["active"]], tSteps=T_STEPS)
    rooms._write_json(os.path.join(data_dir, rooms.NEURAL_META), meta)
    return report, trial


class NeuralModel:
    def __init__(self, data_dir):
        with open(os.path.join(data_dir, rooms.NEURAL_META)) as f:
            meta = json.load(f)
        if meta["labels"] != rooms.LABELS:
            raise ValueError("room list changed since training — retrain")
        self.active = np.array(meta["active"])
        self.node = meta["node"]
        self.temperature = float(meta["temperature"])
        self.hedge = float(meta["hedge"])
        self.steps = int(meta["tSteps"])
        self.net = CSINet(n_cls=len(rooms.LABELS))
        self.net.load_state_dict(torch.load(os.path.join(data_dir, rooms.NEURAL_FILE),
                                            map_location="cpu"))
        self.net.eval()

    def proba(self, A):
        G = rooms.gain_normalise(A, self.active)
        if len(G) < rooms.MIN_FRAMES:
            return None
        W = G[np.linspace(0, len(G) - 1, self.steps).round().astype(int)][None].astype(np.float32)
        return rooms.hedge(softmax_from_logits(_logits(self.net, W), self.temperature)[0], self.hedge)
