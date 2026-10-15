"""
Train the neural CSI classifier on the sessions you recorded with record.py.

  python ai/train.py                 (uses ai/data, writes ai/model.pt)
  python ai/train.py --epochs 50

Once model.pt exists, SGP CSI WIFI automatically switches the AI stage to NEURAL
mode (the Sensor panel shows "neural" instead of "rules").
"""
import argparse, glob, os
import numpy as np
import torch
import torch.nn as nn
from model import CSINet, CLASSES

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "data"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()

    Xs, Ys = [], []
    for f in sorted(glob.glob(os.path.join(a.data, "*.npz"))):
        d = np.load(f, allow_pickle=True)
        lab = str(d["label"])
        if lab not in CLASSES:
            print("skip unknown label:", lab); continue
        w = d["windows"]
        Xs.append(w); Ys += [CLASSES.index(lab)] * len(w)
        print(f"  {os.path.basename(f)}: {len(w)} windows  [{lab}]")
    if not Xs:
        print("No data in", a.data, "— record some with record.py first."); return

    S = max(x.shape[2] for x in Xs)
    X = np.concatenate([np.pad(x, ((0, 0), (0, 0), (0, S - x.shape[2]))) for x in Xs]).astype(np.float32)
    Y = np.array(Ys, dtype=np.int64)
    # per-window normalisation (matches the engine's inference)
    X = (X - X.mean(axis=(1, 2), keepdims=True)) / (X.std(axis=(1, 2), keepdims=True) + 1e-6)
    Xt = torch.tensor(X); Yt = torch.tensor(Y)
    idx = torch.randperm(len(Xt)); Xt, Yt = Xt[idx], Yt[idx]
    nval = max(1, len(Xt) // 5)
    counts = np.bincount(Y, minlength=len(CLASSES))
    print("samples per class:", dict(zip(CLASSES, counts.tolist())), "| total", len(Xt))

    m = CSINet(); opt = torch.optim.Adam(m.parameters(), a.lr); lossf = nn.CrossEntropyLoss()
    best = 0.0
    for ep in range(a.epochs):
        m.train(); perm = torch.randperm(len(Xt) - nval) + nval
        for i in range(0, len(perm), 32):
            b = perm[i:i + 32]
            opt.zero_grad(); l = lossf(m(Xt[b]), Yt[b]); l.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            acc = (m(Xt[:nval]).argmax(1) == Yt[:nval]).float().mean().item()
        best = max(best, acc)
        print(f"epoch {ep + 1:2d}/{a.epochs}  val_acc={acc:.3f}")
    torch.save(m.state_dict(), os.path.join(os.path.dirname(__file__), "model.pt"))
    print(f"Saved model.pt (best val_acc {best:.3f}). Restart the program → NEURAL mode.")

if __name__ == "__main__":
    main()
