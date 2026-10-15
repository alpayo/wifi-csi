#!/usr/bin/env python3
"""Simulated SGP SGPCard card — for testing the PC app without hardware.
Listens for pings on UDP :5566 and replies with synthetic CSI containing a
breathing + moving 'person'. Run alongside:  python3 sgpcard.py --node 127.0.0.1
"""
import math, socket, struct, time, random
HDR = 26; NSUB = 64
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(("0.0.0.0", 5566)); s.settimeout(0.5)
print("sim card listening on :5566 — ping me from sgpcard.py")
# stable baseline channel
base = [random.uniform(10, 40) for _ in range(NSUB)]
active = sorted(random.sample(range(4, NSUB-4), 40))
seq = 0; t0 = time.time()
peer = None
last_emit = 0
last_ap = 0
# fake surrounding WiFi networks (bssid, ssid, base rssi dBm, how much it jitters)
FAKE_APS = [
    (bytes([0x3c,0x84,0x6a,0x11,0x22,0x33]), "MOVISTAR_A4F2", -48, 1),
    (bytes([0xa0,0xbb,0x3e,0x44,0x55,0x66]), "Vodafone-7721", -61, 5),
    (bytes([0xe8,0xde,0x27,0x77,0x88,0x99]), "O2-WiFi-2G",    -70, 2),
    (bytes([0x10,0x2c,0x6b,0xaa,0xbb,0xcc]), "MiFibra-3B1C",  -75, 6),
    (bytes([0x00,0x1a,0x2b,0xdd,0xee,0xff]), "DIGIFIBRA-88",  -82, 3),
]
while True:
    try:
        data, addr = s.recvfrom(64); peer = addr
    except socket.timeout:
        pass
    if not peer:
        continue
    now = time.time()
    if now - last_emit < 0.01:   # ~100 Hz
        continue
    last_emit = now
    t = now - t0
    # a person present after 8s: breathing 15 bpm + slow approach + jitter
    present = t > 8
    breath = math.sin(2*math.pi*0.25*t) if present else 0        # 0.25 Hz = 15 bpm
    rng = 3.2 - 0.25*math.sin(0.05*t) if present else 0
    motion = (0.6 if (present and (int(t)%7==0)) else 0.0)
    buf = bytearray()
    for k in range(NSUB):
        amp = base[k]
        if present and k in active:
            resp = 1.0 - abs((k-active[len(active)//2]))/40.0
            amp += 6*breath*resp + random.gauss(0, 0.6) + motion*8*resp*random.random()
        else:
            amp += random.gauss(0, 0.4)
        amp = max(0, amp)
        ph = random.uniform(-math.pi, math.pi)
        I = int(max(-127, min(127, amp*math.cos(ph))))
        Q = int(max(-127, min(127, amp*math.sin(ph))))
        buf += struct.pack("<bb", I, Q)
    rssi = -48 - int(8*(rng/3)) if present else -55
    mic = (int(300+200*abs(breath)), 120, int(200*abs(breath)), 60)
    pkt = bytearray(HDR)
    pkt[0]=ord('S'); pkt[1]=ord('G'); pkt[2]=1; pkt[3]=0
    struct.pack_into("<II", pkt, 4, seq, int(t*1000)); seq+=1
    struct.pack_into("<bb", pkt, 12, rssi, -92)
    pkt[14]=6; pkt[15]=1
    struct.pack_into("<HHHH", pkt, 16, *mic)
    struct.pack_into("<H", pkt, 24, len(buf))
    s.sendto(bytes(pkt)+bytes(buf), peer)

    # periodically send the WiFi scan ("SA" frame)
    if now - last_ap > 0.7:
        last_ap = now
        ap = bytearray([ord('S'), ord('A'), len(FAKE_APS)])
        for bssid, ssid, r0, jit in FAKE_APS:
            # a person moving adds interference (RSSI jitter) to some links
            r = r0 + random.gauss(0, jit) + (motion * jit * random.random() if present else 0)
            sb = ssid.encode()[:32]
            ap += bssid + struct.pack("<b", int(max(-100, min(-20, r)))) + bytes([6, len(sb)]) + sb
        s.sendto(bytes(ap), peer)
