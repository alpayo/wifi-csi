"""
Train the House map's room classifier on the rooms recorded from the page
(web/rooms.html → Calibrate). The page's Train button does exactly this; the
command line is handy after installing torch, or to read the full report.

  python ai/train_rooms.py                  (uses ai/data/rooms)
  python ai/train_rooms.py --no-neural      (numpy k-NN only, even with torch)

With torch installed it also tries the neural room CSINet and keeps it only if
it beats the numpy k-NN on held-out recordings. Restart SGP CSI WIFI (or press
Train on the page) to load a model trained here.
"""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai import rooms  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "data", "rooms"))
    ap.add_argument("--no-neural", action="store_true")
    a = ap.parse_args()
    try:
        rep = rooms.train(a.data, progress=lambda m: print("  " + m),
                          neural=False if a.no_neural else None)
    except ValueError as e:
        print("Can't train yet:", e); sys.exit(1)
    print("\nmode %s · card %s · %d windows from %d recordings" % (
        rep["mode"], rep["node"], rep["n"], rep["recordings"]))
    merged = ("everything is one zone" if len(rep["zones"]) == 1
              else "after merging zones %.0f%%" % (rep["zoneAccuracy"] * 100))
    print("accuracy %.0f%% (chance %.0f%%) · %s · measured on %s" % (
        rep["accuracy"] * 100, rep["chance"] * 100, merged, rep["scheme"]))
    for c in rep["perClass"]:
        print("  %-14s %s" % (c["name"], "—" if c["recall"] is None else "%3.0f%%" % (c["recall"] * 100)))
    for z in rep["zones"]:
        if len(z["members"]) > 1:
            print("  merged: " + " + ".join(rooms.NAMES[l] for l in z["members"]))
    t = rep.get("neuralTrial")
    if t and t.get("error"):
        print("neural CSINet failed (%s) → k-NN kept" % t["error"])
    elif t:
        print("neural CSINet %.0f%% vs k-NN %.0f%% held out → %s kept" % (
            t["neuralAcc"] * 100, t["knnAcc"] * 100, "neural" if t["kept"] else "k-NN"))


if __name__ == "__main__":
    main()
