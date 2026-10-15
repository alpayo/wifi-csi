"""
Record labelled CSI windows from your SGP Card Mini, to train the neural model.

Connect your PC to the card's WiFi, then run one line per behaviour, acting it
out for the whole duration:

  python ai/record.py --label empty    --secs 40     (leave the room)
  python ai/record.py --label still    --secs 40     (stand still in view)
  python ai/record.py --label walking  --secs 60     (walk around)
  python ai/record.py --label fast     --secs 40     (wave arms / move fast)
  python ai/record.py --label fall     --secs 30     (simulate falls, safely)

Repeat a few times each. Then train:  python ai/train.py
Labels must be one of: empty, still, walking, fast, fall.
"""
import argparse, os, socket, struct, time
import numpy as np

HDR = 26
LABELS = ["empty", "still", "walking", "fast", "fall"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, choices=LABELS)
    ap.add_argument("--node", default="192.168.4.1")
    ap.add_argument("--secs", type=float, default=40)
    ap.add_argument("--win", type=int, default=48)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "data"))
    a = ap.parse_args()
    ip, _, p = a.node.partition(":"); addr = (ip, int(p) if p else 5566)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.bind(("0.0.0.0", 0)); s.settimeout(0.3)
    os.makedirs(a.out, exist_ok=True)
    frames = []; t0 = time.time(); lastping = 0
    print(f"Recording '{a.label}' for {a.secs:.0f}s — act it out now…")
    while time.time() - t0 < a.secs:
        now = time.time()
        if now - lastping > 0.01:
            try: s.sendto(b"SC" + bytes([78]), addr)
            except OSError: pass
            lastping = now
        try: data, _ = s.recvfrom(1024)
        except (socket.timeout, OSError): continue
        if len(data) < HDR or data[0] != ord("S") or data[1] != ord("G"): continue
        clen = struct.unpack_from("<H", data, 24)[0]; buf = data[HDR:HDR + clen]
        n = len(buf) // 2
        if n < 8: continue
        iq = np.frombuffer(bytes(buf[:n * 2]), dtype=np.int8).astype(np.float32)
        frames.append(np.sqrt(iq[0::2] ** 2 + iq[1::2] ** 2))
        if len(frames) % 200 == 0: print(f"  {len(frames)} frames…", end="\r")
    if len(frames) < a.win:
        print("\nToo few frames — is the card streaming and are you on its WiFi?"); return
    S = max(len(f) for f in frames)
    F = np.array([np.pad(f, (0, S - len(f))) for f in frames], dtype=np.float32)
    W = np.array([F[i:i + a.win] for i in range(0, len(F) - a.win, a.win // 2)], dtype=np.float32)
    fn = os.path.join(a.out, f"{a.label}_{int(t0)}.npz")
    np.savez_compressed(fn, windows=W, label=a.label)
    print(f"\nSaved {len(W)} windows ({S} subcarriers) -> {fn}")

if __name__ == "__main__":
    main()
