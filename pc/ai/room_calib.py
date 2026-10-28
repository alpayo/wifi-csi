"""
SGP CSI WIFI — guided room calibration, driven from the House map page.

The page names a room, counts down, and the server records the card's CSI for a
fixed time while you move around that room; then you go to the next one. Every
room (and "nobody moving") is recorded in two passes, so accuracy can be
measured on a recording the model never saw. Each recording is saved the moment
it finishes, so a restart loses nothing and the walk resumes where it stopped;
any single room/pass can be re-recorded without touching the others. Training
runs in the background and the RoomEngine picks the new model up when it ends.
"""
import os, threading, time
import numpy as np
from . import rooms

PASSES = 2
REC_S = 30.0          # seconds recorded per room and pass
COUNTDOWN_S = 5.0     # get-ready time after pressing Start
WALK_S = 10.0         # rough time to walk to the next room (for the estimate)
GAP_S = 2.0           # no CSI for this long aborts a recording

HINTS = {
    rooms.NOBODY: "Nobody moves: everyone stays completely still, or steps outside the flat, "
                  "for the whole recording.",
    "bedroom": "Walk around the whole bedroom (the card's room), into the corners, waving your arms.",
    "entrance": "Walk around the entrance and up and down the hallway, waving your arms.",
    "kitchen": "Walk around the kitchen and the balcony behind the glass, waving your arms.",
}


def hint(label):
    return HINTS.get(label, "Walk around all of %s, into the corners, waving your arms."
                     % rooms.NAMES[label])


class RoomCalibration:
    def __init__(self, data_dir, engine, clock=time.time):
        self.data_dir = data_dir
        self.engine = engine
        self.clock = clock
        os.makedirs(rooms.recordings_dir(data_dir), exist_ok=True)
        self._lock = threading.RLock()
        self.active = None
        self.error = None
        self.train_state = dict(state="idle", msg="", at=None)
        self._scan()

    # ---- what's on disk ----
    def _scan(self):
        have = {}
        d = rooms.recordings_dir(self.data_dir)
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".npz"):
                continue
            try:
                z = np.load(os.path.join(d, fn), allow_pickle=False)
                lab, p = str(z["label"]), int(z["pass_no"])
                t, started = z["t"], float(z["started"])
                if have.get(lab, {}).get(p, {}).get("at", -np.inf) >= started:
                    continue
                have.setdefault(lab, {})[p] = dict(secs=round(float(t[-1] - t[0]), 1),
                                                   frames=int(len(t)), at=started,
                                                   node=str(z["node"]), file=fn)
            except Exception:
                continue
        self.have = have

    def next_step(self):
        """First missing recording in walk order: all of pass 1, then pass 2."""
        for p in range(1, PASSES + 1):
            for lab in rooms.LABELS:
                if p not in self.have.get(lab, {}):
                    return dict(label=lab, pass_no=p)
        return None

    def status(self, now=None):
        now = self.clock() if now is None else now
        with self._lock:
            act = None
            if self.active:
                a = self.active
                if a["phase"] == "countdown":
                    left, prog = a["t_rec"] - now, 0.0
                else:
                    left = a["t_end"] - now
                    prog = min(1.0, max(0.0, (now - a["t_rec"]) / REC_S))
                act = dict(label=a["label"], name=rooms.NAMES[a["label"]], pass_no=a["pass_no"],
                           phase=a["phase"], left=round(max(0.0, left), 1), progress=round(prog, 3),
                           frames=sum(len(t) for t in a["t"]), hint=hint(a["label"]))
            n = len(rooms.LABELS) * PASSES
            trained = (self.engine.report or {}).get("trainedAt")
            newest = max((r["at"] for v in self.have.values() for r in v.values()), default=None)
            return dict(
                classes=[dict(label=l, name=rooms.NAMES[l], hint=hint(l)) for l in rooms.LABELS],
                passes=PASSES, recS=REC_S, countdownS=COUNTDOWN_S,
                totalMin=round(n * (REC_S + COUNTDOWN_S + WALK_S) / 60.0),
                have={l: {str(p): dict(secs=r["secs"], frames=r["frames"], at=r["at"])
                          for p, r in v.items()} for l, v in self.have.items()},
                done=sum(len(v) for v in self.have.values()), total=n,
                canTrain=all(l in self.have for l in rooms.LABELS),
                modelStale=bool(trained and newest and newest > trained),
                next=self.next_step(), active=act, error=self.error,
                train=dict(self.train_state))

    # ---- commands from the page ----
    def handle(self, msg):
        act = msg.get("action")
        if act == "record":
            self.start(msg.get("label"), msg.get("pass"))
        elif act == "cancel":
            self.cancel()
        elif act == "train":
            self.train()
        elif act == "discard":
            self.discard_all()

    def start(self, label, pass_no):
        with self._lock:
            try:
                pass_no = int(pass_no)
            except (TypeError, ValueError):
                pass_no = 0
            if label not in rooms.LABELS or not 1 <= pass_no <= PASSES:
                self.error = "unknown room or pass"
                return False
            if self.active:
                self.error = "a recording is already running"
                return False
            if self.train_state["state"] == "running":
                self.error = "training is running — record when it finishes"
                return False
            now = self.clock()
            self.error = None
            self.active = dict(label=label, pass_no=pass_no, phase="countdown",
                               t_rec=now + COUNTDOWN_S, t_end=now + COUNTDOWN_S + REC_S,
                               node=None, seq=0, t=[], A=[], r=[])
            return True

    def cancel(self):
        with self._lock:
            self.active = None

    def discard_all(self):
        """Start over: every recording and the trained model."""
        with self._lock:
            if self.active or self.train_state["state"] == "running":
                self.error = "stop the recording or wait for training before discarding"
                return
            d = rooms.recordings_dir(self.data_dir)
            for fn in os.listdir(d):
                if fn.endswith(".npz"):
                    os.remove(os.path.join(d, fn))
            for fn in (rooms.KNN_FILE, rooms.REPORT_FILE, rooms.NEURAL_FILE, rooms.NEURAL_META):
                if os.path.exists(os.path.join(self.data_dir, fn)):
                    os.remove(os.path.join(self.data_dir, fn))
            self.error = None
            self.train_state = dict(state="idle", msg="", at=None)
            self._scan()
        self.engine.reload()

    # ---- recording state machine (called from the DSP loop) ----
    def update(self, eng, now=None):
        now = self.clock() if now is None else now
        with self._lock:
            a = self.active
            if not a:
                return
            if a["phase"] == "countdown":
                if now < a["t_rec"]:
                    return
                if eng is None:
                    return self._abort("no card connected — nothing was recorded")
                a.update(phase="recording", node=eng.card, seq=eng.room_buf.seq, last=now)
            if eng is None or eng.card != a["node"]:
                return self._abort("the card went away during the recording — nothing was saved")
            a["seq"], t, A, r = eng.room_buf.since(a["seq"])
            if len(t):
                a["t"].append(t); a["A"].append(A); a["r"].append(r)
                a["last"] = now
            if now - a["last"] > GAP_S:
                return self._abort("CSI stopped during the recording (is this computer still "
                                   "on the card's WiFi?) — nothing was saved")
            if now >= a["t_end"]:
                self._finish()

    def _abort(self, why):
        self.active = None
        self.error = why

    def _finish(self):
        a = self.active
        self.active = None
        t = np.concatenate(a["t"]) if a["t"] else np.zeros(0)
        if len(t) < 10 * REC_S or t[-1] - t[0] < 0.8 * REC_S:
            self.error = ("only %d CSI frames in %.0f s — too few to learn from, nothing was saved"
                          % (len(t), REC_S))
            return
        rooms.save_recording(self.data_dir, a["label"], a["pass_no"], a["node"], t,
                             np.concatenate(a["A"]), np.concatenate(a["r"]))
        self.error = None
        self._scan()

    # ---- training (background thread) ----
    def train(self):
        with self._lock:
            if self.active:
                self.error = "finish or cancel the recording first"
                return False
            if self.train_state["state"] == "running":
                return False
            self.error = None
            self.train_state = dict(state="running", msg="starting", at=self.clock())
        threading.Thread(target=self._train_job, daemon=True).start()
        return True

    def _progress(self, msg):
        with self._lock:
            self.train_state["msg"] = msg

    def _train_job(self):
        try:
            rep = rooms.train(self.data_dir, progress=self._progress)
            self.engine.reload()
            msg = "%s · %.0f%% accuracy (chance %.0f%%) on %s" % (
                "neural" if rep["mode"] == "neural" else "numpy k-NN",
                rep["accuracy"] * 100, rep["chance"] * 100, rep["scheme"])
            state = dict(state="done", msg=msg, at=self.clock())
        except ValueError as e:
            state = dict(state="error", msg=str(e), at=self.clock())
        except Exception as e:                         # keep the server alive, say why
            state = dict(state="error", msg="training failed: %r" % (e,), at=self.clock())
        with self._lock:
            self.train_state = state
