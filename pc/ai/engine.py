"""
SGP CSI WIFI — AI analysis engine.

Runs on every CSI window and returns the person's state + probabilities.

Two modes, chosen automatically:
  • NEURAL  — if a trained model (ai/model.pt) is present, the CSINet classifies
              the raw CSI window (empty/still/walking/fast/fall).
  • RULES   — otherwise it bootstraps the same labels from the features we already
              compute (presence/motion/breathing). Works immediately, no training.

Train the neural model on YOUR hardware with record.py + train.py for accuracy.
Torch is optional: if it is not installed, the engine still runs in RULES mode.
"""
import os
import numpy as np

CLASSES = ["empty", "still", "walking", "fast", "fall"]
_torch = None
_model = None

def _load():
    global _torch, _model
    try:
        import torch
        _torch = torch
        from .model import CSINet
        p = os.path.join(os.path.dirname(__file__), "model.pt")
        if os.path.exists(p):
            m = CSINet()
            m.load_state_dict(torch.load(p, map_location="cpu"))
            m.eval()
            _model = m
    except Exception:
        _torch = None
        _model = None


class AIEngine:
    def __init__(self):
        _load()
        self.model_loaded = _model is not None
        self.has_torch = _torch is not None
        self.smooth = np.ones(len(CLASSES)) / len(CLASSES)
        self._fallArm = 0.0
        self._lastMot = 0.0

    def infer(self, window, feats):
        """window: (T, S) cleaned CSI amplitude; feats: presence/motion/bpm dict."""
        probs = None
        if _model is not None and window is not None and window.shape[0] >= 16:
            try:
                W = window[-64:] if window.shape[0] > 64 else window
                W = (W - W.mean()) / (W.std() + 1e-6)   # same normalisation as training
                x = _torch.tensor(W[None].astype("float32"))
                with _torch.no_grad():
                    probs = _torch.softmax(_model(x), 1).numpy()[0]
                source = "neural"
            except Exception:
                probs = None
        if probs is None:
            probs = self._rules(feats)
            source = "rules"
        self.smooth = 0.72 * self.smooth + 0.28 * probs
        i = int(np.argmax(self.smooth))
        return dict(activity=CLASSES[i], conf=round(float(self.smooth[i]), 2),
                    probs=[round(float(x), 2) for x in self.smooth],
                    source=source, model=self.model_loaded, classes=CLASSES)

    def _rules(self, f):
        pres = float(f.get("presence", 0.0))
        mot = float(f.get("motion", 0.0))
        # crude fall cue: a big motion burst that collapses to stillness
        fall = 0.0
        if self._lastMot > 0.6 and mot < 0.2 and pres > 0.5:
            self._fallArm = 1.0
        self._fallArm *= 0.85
        fall = self._fallArm if self._fallArm > 0.15 else 0.0
        self._lastMot = mot
        p = np.zeros(len(CLASSES))
        walk = min(mot * 3.0, 1.0) * (1.0 - min(max(mot - 0.5, 0) * 2, 1.0))
        p[0] = max(0.0, 1.0 - pres * 1.2)          # empty
        p[1] = pres * (1.0 - min(mot * 3.0, 1.0))  # still
        p[2] = pres * walk                          # walking
        p[3] = pres * min(max(mot - 0.5, 0) * 2, 1.0)  # fast
        p[4] = fall
        s = p.sum() + 1e-6
        return p / s
