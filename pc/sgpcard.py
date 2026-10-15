#!/usr/bin/env python3
"""
SGP SGPCard Command — PC side of the SGP Card Mini WiFi-sensing system.

The ESP32 card is a dumb sensor: it streams raw WiFi CSI (+ mic energy) over
UDP. THIS program does all the work — signal processing, the presence/motion/
breathing algorithms and the "AI", range estimation, optional multi-node
trilateration — and serves a tactical 3D view in the browser.

Pipeline (per node):
  raw CSI I/Q  ->  per-subcarrier amplitude/phase
               ->  Hampel outlier clean + active-subcarrier select
               ->  online ambient baseline (self-calibrating)
               ->  motion energy + presence score (phase-variance detector)
               ->  breathing band (0.1-0.5 Hz) -> BPM via FFT
               ->  range estimate (CSI disturbance + RSSI, Fresnel-scaled)
  N nodes with known positions -> least-squares trilateration -> real (x,y,z)

Run:   python3 sgpcard.py            (card at 192.168.4.1, default)
       python3 sgpcard.py --node 192.168.4.1:5566
       python3 sgpcard.py --node A=192.168.4.1 --node B=192.168.1.51   (multi)
"""
import argparse, asyncio, http.server, json, math, socket, struct, threading, time, webbrowser
from collections import deque
import numpy as np

# optional AI analysis stage (neural model if trained, rules otherwise; no torch = rules only)
try:
    from ai.engine import AIEngine
except Exception:
    AIEngine = None

HDR = 26
PORT = 8799          # web + websocket port on the PC
CARD_PORT = 5566     # UDP port the cards listen on (pings + discovery)
PING_HZ = 100        # how fast we poke the card to generate CSI (overridden by CONFIG)
WIN = 1024           # rolling CSI frames kept per node (~12 s @ 80 Hz)
BASELINE_S = 6.0     # ambient self-calibration time
SPEED_OF_LIGHT = 3e8

# Live, user-adjustable settings (changed from the Settings panel over the websocket)
CONFIG = {
    'pingHz':    100,    # how fast the PC pokes the card (Turbo can push this to ~2000)
    'txPower':   78,     # card WiFi TX power (0.25 dBm units, 8..84) -> sent to the card
    'presenceBias': 1.8, # lower = more sensitive presence
    'motionGain':   6.0, # motion display gain
    'sensitivity':  1.0, # 1 = normal; higher = the tiniest change above the noise floor shows
    'smooth':       0.6, # output smoothing (0..0.95)
    'rangeCalib':   0.0, # metres added to the range estimate
    'useAnchors':   True,# use room WiFi networks to triangulate bearing
    'losMode':      False,# 2-Meter Mode: person on the straight card<->PC line
    'losDist':      2.0, # metres between the device and the computer
}
# WiFi "anchors": the user picks which networks to trust and where they sit.
#   bssid(str) -> {'sel':bool, 'angle':rad}
AP_SEL = {}
# distance from RSSI via a simple log-distance path-loss model
def rssi_to_dist(rssi, rssi0=-40.0, n=2.6):
    return float(min(30.0, max(0.4, 10 ** ((rssi0 - rssi) / (10.0 * n)))))

# ──────────────────────────────────────────────────────────────────────────
# Per-node sensing engine
# ──────────────────────────────────────────────────────────────────────────
class NodeEngine:
    def __init__(self, name, pos=(0.0, 0.0, 0.0), yaw=0.0):
        self.name = name
        self.pos = np.array(pos, float)   # node position in the room (metres)
        self.yaw = yaw                    # boresight heading (rad), +x = 0
        self.amp = deque(maxlen=WIN)      # amplitude vectors
        self.ts = deque(maxlen=WIN)       # timestamps (s)
        self.rssi = deque(maxlen=WIN)
        self.mic = (0, 0, 0, 0)
        self.active = None                # indices of usable subcarriers
        self.base_mu = None               # ambient baseline mean per subcarrier
        self.base_sd = None
        self.base_t0 = None
        self.calibrated = False
        self.last_rx = 0.0
        self.count = 0
        self.aps = {}   # bssid -> {'ssid','ch','rssi':deque,'last':t}
        self.ai = AIEngine() if AIEngine else None
        # smoothed outputs
        self.s_motion = 0.0
        self.s_presence = 0.0
        self.s_range = 2.0
        self.s_bpm = 0.0
        self.bpm_conf = 0.0
        self.mot_mu = None        # adaptive mean of the quiet-channel motion metric
        self.mot_var = 0.0        # ...and its variance (the noise model)

    # ---- ingest one CSI frame ----
    def feed(self, t_ms, rssi, noise, buf, mic, first_bad):
        n = len(buf) // 2
        if n < 8:
            return
        iq = np.frombuffer(bytes(buf[: n * 2]), dtype=np.int8).astype(np.float32)
        I, Q = iq[0::2], iq[1::2]
        a = np.sqrt(I * I + Q * Q)
        if first_bad and len(a) > 2:        # first LTF word can be garbage
            a[0] = a[1] = 0
        self.amp.append(a)
        self.ts.append(time.time())
        self.rssi.append(rssi)
        self.mic = mic
        self.last_rx = time.time()
        self.count += 1

    # ---- ingest a WiFi scan: list of (bssid, ssid, rssi, ch) ----
    def feedAPs(self, aps):
        now = time.time()
        for bssid, ssid, rssi, ch in aps:
            a = self.aps.get(bssid)
            if a is None:
                a = {'ssid': ssid, 'ch': ch, 'rssi': deque(maxlen=40), 'last': now}
                self.aps[bssid] = a
            if ssid:
                a['ssid'] = ssid
            a['ch'] = ch; a['last'] = now
            a['rssi'].append(rssi)
        # forget APs not heard for a while
        for k in [k for k, v in self.aps.items() if now - v['last'] > 20]:
            del self.aps[k]

    def apList(self):
        """Current APs with RSSI, distance and 'interference' (RSSI variance)."""
        now = time.time()
        out = []
        for bssid, a in self.aps.items():
            r = np.array(a['rssi'], float)
            rssi = float(r[-1]) if len(r) else -100.0
            var = float(np.std(r)) if len(r) > 4 else 0.0
            sel = AP_SEL.get(bssid, {})
            out.append(dict(
                bssid=bssid, ssid=a['ssid'] or '(hidden)', ch=a['ch'],
                rssi=round(rssi, 1), dist=round(rssi_to_dist(rssi), 2),
                interf=round(min(1.0, var / 6.0), 3),
                sel=bool(sel.get('sel', False)), angle=round(float(sel.get('angle', 0.0)), 3),
                age=round(now - a['last'], 1)))
        out.sort(key=lambda x: x['rssi'], reverse=True)
        return out

    # ---- Hampel-style robust clean of the latest matrix ----
    @staticmethod
    def _clean(M):
        med = np.median(M, axis=0)
        mad = np.median(np.abs(M - med), axis=0) + 1e-6
        bad = np.abs(M - med) > 5 * 1.4826 * mad
        Mc = M.copy()
        # replace outliers column-wise with the column median
        for j in range(M.shape[1]):
            col = Mc[:, j]
            col[bad[:, j]] = med[j]
        return Mc

    # ---- run DSP on the current window, return metrics ----
    def process(self):
        if len(self.amp) < 24:
            return None
        L = min(len(self.amp), WIN)
        M = np.array([self.amp[i] for i in range(len(self.amp) - L, len(self.amp))], float)
        ts = np.array([self.ts[i] for i in range(len(self.ts) - L, len(self.ts))], float)

        # choose active subcarriers once (non-zero, stable energy)
        if self.active is None and L > 40:
            energy = np.mean(M, axis=0)
            self.active = np.where(energy > 0.15 * np.max(energy))[0]
            if len(self.active) < 4:
                self.active = np.arange(M.shape[1])
        act = self.active if self.active is not None else np.arange(M.shape[1])
        A = self._clean(M[:, act])

        # online ambient baseline (self-calibration during the first seconds)
        if self.base_t0 is None:
            self.base_t0 = ts[0]
        if not self.calibrated:
            if ts[-1] - self.base_t0 >= BASELINE_S and L > 60:
                self.base_mu = np.median(A, axis=0)
                self.base_sd = np.median(np.abs(A - self.base_mu), axis=0) * 1.4826 + 1e-3
                self.calibrated = True
        mu = self.base_mu if self.calibrated else np.median(A, axis=0)
        sd = self.base_sd if self.calibrated else (np.std(A, axis=0) + 1e-3)

        # --- motion energy: frame-to-frame change over recent frames ---
        recent = A[-min(24, L):]
        dif = np.abs(np.diff(recent, axis=0))
        raw = float(np.mean(dif) / (np.mean(mu) + 1e-3))
        # Adaptive noise model of the QUIET channel (running mean + variance),
        # learned only while we're near baseline so a real movement never teaches
        # the floor to ignore it.
        if self.mot_mu is None:
            self.mot_mu = raw; self.mot_var = (raw * 0.15) ** 2 + 1e-9
        sigma = math.sqrt(self.mot_var)
        if raw < self.mot_mu + 1.5 * sigma:          # learn ONLY while genuinely quiet
            d = raw - self.mot_mu
            self.mot_mu += 0.006 * d                 # slow: don't absorb a real signal
            self.mot_var = 0.994 * self.mot_var + 0.006 * d * d
        # How many sigma above the quiet floor is this frame? Sensitivity sets the
        # sigma that counts as "full motion": crank it up and a tiny excess (a hand
        # near the device) already reads as movement, while ambient noise (~1 sigma)
        # stays quiet → it won't false-trigger at max sensitivity.
        z = (raw - self.mot_mu) / (sigma + 1e-9)
        sens = float(CONFIG.get('sensitivity', 1.0))
        thr = max(1.5, 14.0 / max(1.0, sens))        # sens 1 -> 14σ, sens 50 -> ~1.5σ
        motion = max(0.0, min(z / thr, 1.0))

        # --- presence score: deviation from ambient baseline (z-scored) ---
        dev = np.abs(A[-min(16, L):] - mu) / sd
        presence_raw = float(np.mean(dev))
        # logistic squashing into 0..1 (the "presence head"; phase-variance fallback)
        presence = 1.0 / (1.0 + math.exp(-(presence_raw - CONFIG['presenceBias']) * 1.3))

        # --- breathing: strongest subcarrier in 0.1-0.5 Hz band ---
        bpm, bconf = self._breathing(A, ts)

        # --- range estimate (single node, Fresnel/RSSI heuristic) ---
        rng = self._range(presence_raw, motion)

        # smooth (smoothing + gains are live-adjustable from the Settings panel)
        sm = float(CONFIG['smooth'])
        self.s_motion = sm * self.s_motion + (1 - sm) * min(motion * (CONFIG['motionGain'] / 6.0), 1.0)
        self.s_presence = (sm * 0.9) * self.s_presence + (1 - sm * 0.9) * presence
        if rng is not None:
            self.s_range = 0.85 * self.s_range + 0.15 * (rng + CONFIG['rangeCalib'])
        if bconf > 0.25:
            self.s_bpm = 0.8 * self.s_bpm + 0.2 * bpm
            self.bpm_conf = bconf
        else:
            self.bpm_conf *= 0.9

        rssi_now = float(np.mean(list(self.rssi)[-10:])) if self.rssi else -100
        fs = L / max(ts[-1] - ts[0], 1e-3)
        # per-subcarrier response vector for the heatmap
        resp = (np.std(A[-min(40, L):], axis=0) / (mu + 1e-3))
        resp = np.clip(resp / (np.max(resp) + 1e-6), 0, 1)

        # --- spectrogram column: FFT of the aggregate CSI-perturbation signal ---
        spec = []
        specFmax = 5.0
        if L >= 128:
            sig = np.mean(A - mu, axis=1)
            tu = np.linspace(ts[0], ts[-1], L)
            sig = np.interp(tu, ts, sig); sig = sig - np.mean(sig)
            fsr = L / max(ts[-1] - ts[0], 1e-3)
            sp = np.abs(np.fft.rfft(sig * np.hanning(L)))
            fr = np.fft.rfftfreq(L, d=1.0 / fsr)
            band = fr <= specFmax
            sp = sp[band]
            if len(sp) > 2:
                col = np.interp(np.linspace(0, len(sp) - 1, 64), np.arange(len(sp)), sp)
                col = np.log1p(col * 4.0)
                col = col / (np.max(col) + 1e-6)
                spec = [round(float(x), 3) for x in col]

        return dict(
            name=self.name, pos=self.pos.tolist(), yaw=self.yaw,
            presence=round(self.s_presence, 3),
            motion=round(self.s_motion, 3),
            rangeM=round(self.s_range, 2),
            bpm=round(self.s_bpm, 1), bpmConf=round(self.bpm_conf, 2),
            rssi=round(rssi_now, 1), fs=round(fs, 1),
            calibrated=self.calibrated,
            mic=self.mic, nActive=int(len(act)),
            resp=[round(float(x), 3) for x in resp],
            spec=spec, specFmax=specFmax,
            aps=self.apList(),
            ai=self._ai(A),
        )

    def _ai(self, A):
        if self.ai is None:
            return None
        try:
            return self.ai.infer(A, dict(presence=self.s_presence, motion=self.s_motion, bpm=self.s_bpm))
        except Exception:
            return None

    def _breathing(self, A, ts):
        L = A.shape[0]
        dur = ts[-1] - ts[0]
        if L < 128 or dur < 6:
            return 0.0, 0.0
        fs = L / dur
        # resample to uniform grid
        tu = np.linspace(ts[0], ts[-1], L)
        # pick the subcarrier with most 0.1-0.5 Hz energy
        best, bestP, bestBpm = -1, 0.0, 0.0
        sig_all = A - np.mean(A, axis=0)
        # limit work: test up to 24 subcarriers
        cols = range(A.shape[1]) if A.shape[1] <= 24 else np.linspace(0, A.shape[1]-1, 24).astype(int)
        freqs = np.fft.rfftfreq(L, d=1.0 / fs)
        band = (freqs >= 0.1) & (freqs <= 0.6)
        if not np.any(band):
            return 0.0, 0.0
        for j in cols:
            s = np.interp(tu, ts, A[:, j]); s = s - np.mean(s)
            sp = np.abs(np.fft.rfft(s * np.hanning(L)))
            p = np.max(sp[band])
            tot = np.sum(sp[1:]) + 1e-6
            if p > bestP:
                bestP = p; best = j
                bestBpm = freqs[band][np.argmax(sp[band])] * 60.0
                bestProm = p / tot
        conf = float(np.clip(bestProm * 8, 0, 1)) if best >= 0 else 0.0
        return float(bestBpm), conf

    def _range(self, presence_raw, motion):
        # Single-antenna CSI cannot measure true range. We map disturbance
        # magnitude to a PSEUDO-range (bigger disturbance ~ closer), plus RSSI.
        # Honest estimate with wide uncertainty — refined by multi-node below.
        if not self.calibrated:
            return None
        rssi = np.mean(list(self.rssi)[-10:]) if self.rssi else -70
        # disturbance 0..~4 -> range 5..0.4 m
        d = np.clip(presence_raw, 0.2, 5.0)
        rng = 4.6 - 0.62 * d
        # closer stations usually stronger RSSI; nudge
        rng += np.clip((-55 - rssi) * 0.03, -1.0, 1.0)
        return float(np.clip(rng, 0.4, 6.0))


# ──────────────────────────────────────────────────────────────────────────
# Multi-node trilateration (when >=3 nodes see the person)
# ──────────────────────────────────────────────────────────────────────────
def trilaterate(nodes):
    seen = [n for n in nodes if n.calibrated and n.s_presence > 0.4]
    if len(seen) < 3:
        return None
    P = np.array([n.pos for n in seen])
    r = np.array([n.s_range for n in seen])
    # linearized least squares around the first anchor
    x0, y0 = P[0, 0], P[0, 1]
    A, b = [], []
    for i in range(1, len(seen)):
        A.append([2 * (P[i, 0] - x0), 2 * (P[i, 1] - y0)])
        b.append(r[0] ** 2 - r[i] ** 2 - x0 ** 2 - y0 ** 2 + P[i, 0] ** 2 + P[i, 1] ** 2)
    try:
        sol, *_ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
        return [float(sol[0]), float(sol[1]), 1.0]
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────
# UDP: ping each card + receive its CSI
# ──────────────────────────────────────────────────────────────────────────
class NodeLink:
    def __init__(self, engine, ip, port=5566):
        self.engine = engine
        self.addr = (ip, port)
        self.stop = threading.Event()        # per-link, so one card can be removed alone
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", 0))
        self.sock.settimeout(0.2)

    def start(self):
        threading.Thread(target=self.ping_loop, daemon=True).start()
        threading.Thread(target=self.recv_loop, daemon=True).start()

    def close(self):
        self.stop.set()
        try: self.sock.close()
        except Exception: pass

    def ping_loop(self):
        # Each ping the card receives produces one CSI frame, so the ping rate IS
        # the CSI rate. Python's sleep floor is ~1 ms, so to reach Turbo rates we
        # send a small BURST per wake instead of one ping at a time.
        while not self.stop.is_set():
            hz = max(1, min(3000, int(CONFIG['pingHz'])))
            burst = max(1, hz // 500)
            p = b"SC" + bytes([int(CONFIG['txPower']) & 0xFF])   # ping carries TX power
            for _ in range(burst):
                try: self.sock.sendto(p, self.addr)
                except OSError: pass
            time.sleep(burst / hz)

    def recv_loop(self):
        while not self.stop.is_set():
            try:
                data, _ = self.sock.recvfrom(1024)
            except socket.timeout:
                continue
            except OSError:
                continue
            if len(data) < 3 or data[0] != ord("S"):
                continue
            if data[1] == ord("G") and len(data) >= HDR:          # CSI frame
                t_ms = struct.unpack_from("<I", data, 8)[0]
                rssi = struct.unpack_from("<b", data, 12)[0]
                noise = struct.unpack_from("<b", data, 13)[0]
                flags = data[3]
                mic = struct.unpack_from("<HHHH", data, 16)
                clen = struct.unpack_from("<H", data, 24)[0]
                self.engine.feed(t_ms, rssi, noise, data[HDR:HDR + clen], mic, flags & 1)
            elif data[1] == ord("A"):                              # WiFi scan: "SA" + count + entries
                count = data[2]; off = 3; aps = []
                try:
                    for _ in range(count):
                        bssid = ":".join("%02x" % b for b in data[off:off + 6])
                        rssi = struct.unpack_from("<b", data, off + 6)[0]
                        ch = data[off + 7]; slen = data[off + 8]
                        ssid = data[off + 9:off + 9 + slen].decode("utf-8", "replace")
                        off += 9 + slen
                        aps.append((bssid, ssid, rssi, ch))
                    self.engine.feedAPs(aps)
                except Exception:
                    pass


# ──────────────────────────────────────────────────────────────────────────
# Live node registry — cards can be added / placed / removed from the browser
# (multi-card mode). >=3 placed cards that see the person => real trilateration.
# ──────────────────────────────────────────────────────────────────────────
ENGINES, LINKS = [], []
NODE_LOCK = threading.Lock()
# suggested room layout (metres) for the Nth card when the UI doesn't give one
DEFAULT_POS = [(0, 0, 0), (3, 0, 0), (0, 3, 0), (3, 3, 0), (1.5, 3, 0), (3, 1.5, 0)]

def add_node(name=None, ip="192.168.4.1", port=5566, pos=None, mac=None):
    with NODE_LOCK:
        existing = {e.name for e in ENGINES}
        if name:
            name = str(name)[:12]
        else:                                           # first free letter A, B, C…
            i = 0
            while i < 25 and chr(ord('A') + i) in existing:
                i += 1
            name = chr(ord('A') + i)
        if name in existing:                            # guarantee uniqueness
            k = 2
            while f"{name}{k}" in existing:
                k += 1
            name = f"{name}{k}"
        if pos is None:
            pos = DEFAULT_POS[len(ENGINES) % len(DEFAULT_POS)]
        eng = NodeEngine(name, pos=pos)
        eng.mac = mac
        lk = NodeLink(eng, ip, int(port))
        ENGINES.append(eng); LINKS.append(lk)
        lk.start()
        print(f"  + card {name} @ {ip}:{port}  at {tuple(round(float(v),2) for v in eng.pos)}")
        return eng

def remove_node(name):
    with NODE_LOCK:
        for i, e in enumerate(ENGINES):
            if e.name == name:
                LINKS[i].close()
                ENGINES.pop(i); LINKS.pop(i)
                print(f"  - card {name} removed")
                return True
    return False

def set_node_pos(name, x, y):
    for e in ENGINES:
        if e.name == name:
            e.pos = np.array([float(x), float(y), float(e.pos[2])])
            return True
    return False

def _has_ip(ip):
    with NODE_LOCK:
        return any(lk.addr[0] == ip for lk in LINKS)

def _has_mac(mac):
    if not mac:
        return False
    with NODE_LOCK:
        return any(getattr(e, 'mac', None) == mac for e in ENGINES)


# ──────────────────────────────────────────────────────────────────────────
# Auto-detection — find SGP Card Minis on the shared WiFi with zero config.
# The PC sweeps its own /24 (and the AP-mode default) with a tiny "SD" probe;
# every card replies "SGH"+name+mac, and we add any new one by its source IP.
# Unicast works through phone hotspots where broadcast is often blocked.
# ──────────────────────────────────────────────────────────────────────────
DISCOVER = {'on': True, 'last': 0.0, 'found': 0, 'subnet': ''}

def my_lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))     # no packet sent; just selects the egress iface
        ip = s.getsockname()[0]; s.close()
        return ip
    except Exception:
        return None

def discover_loop(stop):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try: sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    except Exception: pass
    sock.bind(("0.0.0.0", 0))
    sock.settimeout(0.3)
    while not stop.is_set():
        if not DISCOVER['on']:
            time.sleep(0.5); continue
        ip = my_lan_ip()
        prefix = ip.rsplit('.', 1)[0] if ip else None
        DISCOVER['subnet'] = (prefix + '.x') if prefix else ''
        # spray the discovery probe: our whole /24 (unicast) + AP-mode default +
        # loopback (a sim on this PC) + broadcast
        targets = ["192.168.4.1", "127.0.0.1"]
        if prefix:
            targets += [f"{prefix}.{i}" for i in range(1, 255) if f"{prefix}.{i}" != ip]
        for t in targets:
            try: sock.sendto(b"SD", (t, CARD_PORT))
            except OSError: pass
        try: sock.sendto(b"SD", ("255.255.255.255", CARD_PORT))
        except OSError: pass
        DISCOVER['last'] = time.time()
        # collect "SGH" replies for a couple of seconds, auto-add new cards
        t0 = time.time()
        while time.time() - t0 < 2.5 and not stop.is_set():
            try:
                data, addr = sock.recvfrom(128)
            except socket.timeout:
                continue
            except OSError:
                break
            if len(data) >= 5 and data[:3] == b"SGH":
                try:
                    L = data[4]; nm = data[5:5 + L].decode("utf-8", "replace")
                    mac = data[5 + L:5 + L + 6].hex() if len(data) >= 5 + L + 6 else None
                except Exception:
                    nm, mac = "", None
                if _has_mac(mac) or _has_ip(addr[0]):
                    continue                               # already have this card
                short = (nm.rsplit('_', 1)[-1] or '')[:10] or None   # "SGP_SGPCARD_1a2b" -> "1a2b"
                add_node(short, addr[0], CARD_PORT, mac=mac)
                DISCOVER['found'] += 1
        time.sleep(1.0)


# ──────────────────────────────────────────────────────────────────────────
# Web / websocket server
# ──────────────────────────────────────────────────────────────────────────
STATE = {"nodes": [], "person": None, "t": 0}

def apply_client_msg(msg):
    """Settings + WiFi-anchor choices coming back from the browser."""
    t = msg.get('type')
    if t == 'cfg':
        for k, v in msg.get('cfg', {}).items():
            if k in CONFIG:
                try: CONFIG[k] = type(CONFIG[k])(v)
                except Exception: pass
    elif t == 'ap':
        b = msg.get('bssid')
        if b:
            a = AP_SEL.setdefault(b, {'sel': False, 'angle': 0.0})
            if 'sel' in msg: a['sel'] = bool(msg['sel'])
            if 'angle' in msg: a['angle'] = float(msg['angle'])
    elif t == 'node':
        act = msg.get('action')
        if act == 'add':
            pos = None
            if 'x' in msg and 'y' in msg:
                pos = (float(msg['x']), float(msg['y']), 0.0)
            add_node(msg.get('name'), msg.get('ip', '192.168.4.1'),
                     int(msg.get('port', 5566)), pos)
        elif act == 'pos':
            set_node_pos(msg.get('name'), msg.get('x', 0), msg.get('y', 0))
        elif act == 'remove':
            remove_node(msg.get('name'))
    elif t == 'discover':
        DISCOVER['on'] = bool(msg.get('on', True))
    elif t == 'recal':
        for e in msg.get('_engines', []):
            pass  # handled below via a flag

RECAL = {'flag': False}

async def ws_recv(ws):
    try:
        async for raw in ws:
            try:
                m = json.loads(raw)
                if m.get('type') == 'recal':
                    RECAL['flag'] = True
                else:
                    apply_client_msg(m)
            except Exception:
                pass
    except Exception:
        pass

async def ws_handler(ws):
    recv_task = asyncio.ensure_future(ws_recv(ws))
    try:
        while True:
            await ws.send(json.dumps(STATE))
            await asyncio.sleep(0.05)
    except Exception:
        pass
    finally:
        recv_task.cancel()

def serve_http(webdir):
    class H(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=webdir, **k)
        def log_message(self, *a):
            pass
        def end_headers(self):
            # never cache — so the browser always gets the latest UI + 3D models
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Pragma", "no-cache")
            super().end_headers()
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), H)
    httpd.serve_forever()

async def ws_main():
    import websockets
    async with websockets.serve(ws_handler, "127.0.0.1", PORT + 1, ping_interval=None):
        await asyncio.Future()


# ──────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", action="append", default=[],
                    help="card address, e.g. 192.168.4.1 or NAME=IP[:port]. Repeat for multi-node.")
    ap.add_argument("--no-discover", action="store_true",
                    help="disable auto-detection of cards on the network")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    DISCOVER['on'] = not args.no_discover
    # Explicit --node always wins. Otherwise, if auto-detect is on we start with
    # NO cards and let discovery find them; if it's off we fall back to the
    # single-card AP default so the app still works out of the box.
    specs = args.node if args.node else ([] if DISCOVER['on'] else ["192.168.4.1"])
    for spec in specs:
        name = None
        if "=" in spec:
            name, spec = spec.split("=", 1)
        ip, _, p = spec.partition(":")
        add_node(name, ip, int(p) if p else 5566)   # registers + starts the link

    stop = threading.Event()

    import os
    webdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
    threading.Thread(target=serve_http, args=(webdir,), daemon=True).start()
    if DISCOVER['on']:
        threading.Thread(target=discover_loop, args=(stop,), daemon=True).start()

    def dsp_loop():
        while not stop.is_set():
            with NODE_LOCK:
                engs = list(ENGINES)          # snapshot — cards may be added/removed live
            # recalibrate the empty-room baseline on request from the UI
            if RECAL['flag']:
                RECAL['flag'] = False
                for e in engs:
                    e.calibrated = False; e.base_t0 = None; e.base_mu = None; e.base_sd = None
                    e.mot_mu = None; e.mot_var = 0.0
            out = []
            for e in engs:
                m = e.process()
                if not m:
                    # card registered but not streaming yet — still show it so it
                    # can be seen and placed in the room (online=False)
                    m = dict(name=e.name, pos=e.pos.tolist(), yaw=float(e.yaw),
                             presence=0.0, motion=0.0, rangeM=0.0, rssi=0,
                             calibrated=bool(e.calibrated), fs=0, nActive=0,
                             bpm=0.0, bpmConf=0.0, spec=[], aps=[],
                             mic=[0, 0, 0, 0], ai=None)
                m["online"] = (time.time() - e.last_rx) < 1.5
                out.append(m)
            person = trilaterate(engs)
            if person is None and out and CONFIG.get('losMode'):
                # 2-METER MODE: the computer (TX) and the card (RX) face each other
                # ~losDist apart; the person stands on that straight line and
                # disturbs the direct link. Place them ON the card->PC axis (+x)
                # at the CSI range estimate, clamped to the baseline length.
                e = engs[0]
                if e.s_presence > 0.30:                 # sensitive: direct LoS crossing
                    D = float(CONFIG.get('losDist', 2.0))
                    r = min(max(e.s_range, 0.1), D)
                    person = [float(e.pos[0] + r), float(e.pos[1]), 1.0]
            elif person is None and out:
                # single node: range from CSI, BEARING from the room's WiFi anchors
                e = engs[0]; m = out[0]
                if e.s_presence > 0.45:
                    r = e.s_range
                    bearing = e.yaw
                    if CONFIG.get('useAnchors', True):
                        xs = ys = wsum = 0.0
                        for a in m.get('aps', []):
                            if a['sel'] and a['interf'] > 0.08:
                                w = a['interf']; wsum += w
                                xs += w * math.cos(a['angle']); ys += w * math.sin(a['angle'])
                        if wsum > 0:
                            bearing = math.atan2(ys, xs)
                    person = [float(e.pos[0] + r * math.cos(bearing)),
                              float(e.pos[1] + r * math.sin(bearing)), 1.0]
            STATE["nodes"] = out
            STATE["person"] = person
            STATE["discover"] = {"on": DISCOVER['on'], "found": DISCOVER['found'],
                                 "subnet": DISCOVER['subnet']}
            STATE["t"] = time.time()
            time.sleep(0.05)
    threading.Thread(target=dsp_loop, daemon=True).start()

    url = f"http://127.0.0.1:{PORT}/index.html"
    print("=" * 60)
    print("  SGP SGPCard Command")
    print("  nodes:", ", ".join(f"{e.name}@{lk.addr[0]}" for e, lk in zip(ENGINES, LINKS)))
    print("  open:", url)
    print("=" * 60)
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        asyncio.run(ws_main())
    except KeyboardInterrupt:
        stop.set()


if __name__ == "__main__":
    main()
