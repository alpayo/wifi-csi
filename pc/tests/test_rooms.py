"""Tests for the House map room classifier, its guided calibration and the page.

Run from the repository root:  python -m unittest discover -s pc/tests
(numpy only; the neural test runs when torch is installed)."""
import asyncio, importlib.util, json, os, shutil, socket, sys, tempfile, time, unittest
from html.parser import HTMLParser
from unittest import mock
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PC = os.path.dirname(HERE)
sys.path.insert(0, PC)
sys.path.insert(0, HERE)

from ai import rooms                                   # noqa: E402
from ai.room_calib import RoomCalibration, COUNTDOWN_S, REC_S, GAP_S   # noqa: E402
from synth import House                                # noqa: E402

NODE = "f908"


def record_house(d, house, passes=2, secs=30.0, node=NODE, t0=1.7e9):
    for p in range(1, passes + 1):
        for lab in rooms.LABELS:
            t, A, r = house.record(lab, secs=secs, t0=t0)
            rooms.save_recording(d, lab, p, node, t, A, r)
            t0 += secs + 15
    return t0


def feed(buf, t, A, r):
    for i in range(len(t)):
        buf.push(t[i], A[i], r[i])


def live_on(data_dir, house, label, t0, secs=4.0):
    """First live output for `label` from a fresh engine (no smoothing history)."""
    buf = rooms.FrameBuffer()
    t, A, r = house.record(label, secs=secs, t0=t0)
    feed(buf, t, A, r)
    return rooms.RoomEngine(data_dir).infer(buf, NODE, now=t[-1])


class TrainedHouse:
    """One synthetic house, recorded and trained once per test class."""
    house_kw = {}

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="rooms-test-")
        cls.house = House(rooms.LABELS, seed=7, **cls.house_kw)
        cls.t_end = record_house(cls.dir, cls.house)
        cls.report = rooms.train(cls.dir, neural=False)
        cls.engine = rooms.RoomEngine(cls.dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)


# ──────────────────────────────────────────────────────────────────────────
class FeatureTests(unittest.TestCase):
    def test_features_ignore_per_frame_gain(self):
        house = House(rooms.LABELS, seed=1)
        t, A, _ = house.record("cem", secs=2.0)
        act = rooms.select_active(A)
        gains = np.random.default_rng(0).uniform(0.5, 1.5, (len(A), 1))
        np.testing.assert_allclose(rooms.window_features(A, act),
                                   rooms.window_features(A * gains, act), atol=1e-5)

    def test_active_subcarriers_skip_the_nulls(self):
        house = House(rooms.LABELS, seed=1)
        _, A, _ = house.record("cem", secs=2.0)
        act = set(rooms.select_active(A).tolist())
        self.assertFalse(act & {0, 1, 27, 32, 37})
        self.assertGreater(len(act), 40)

    def test_frame_buffer_caps_the_rate_by_dropping_frames(self):
        buf = rooms.FrameBuffer()
        t = 1000.0 + np.arange(3000) / 1500.0               # 2 s at Turbo rate
        for ti in t:
            buf.push(ti, np.ones(64), -40)
        self.assertLessEqual(buf.seq, 2 * rooms.MAX_HZ + 1)
        self.assertGreaterEqual(buf.seq, 2 * rooms.MAX_HZ - 1)
        tw, A = buf.window(1.0)
        self.assertTrue(np.all(A == 1.0))                  # dropped, never averaged
        self.assertAlmostEqual(tw[-1] - tw[0], 1.0, delta=0.02)

    def test_frame_buffer_pads_short_frames_and_reports_new_ones(self):
        buf = rooms.FrameBuffer()
        buf.push(5.0, np.full(52, 3.0), -40)
        seq, t, A, r = buf.since(0)
        self.assertEqual((seq, A.shape), (1, (1, 64)))
        self.assertTrue(np.all(A[0, :52] == 3.0) and np.all(A[0, 52:] == 0))
        self.assertEqual(len(buf.since(seq)[1]), 0)


# ──────────────────────────────────────────────────────────────────────────
class CrossValidationTests(unittest.TestCase):
    def test_rooms_recorded_twice_are_tested_one_whole_pass_at_a_time(self):
        # 3 classes x 2 passes, 10 windows per recording
        y = np.repeat([0, 1, 2, 0, 1, 2], 10)
        pas = np.repeat([1, 1, 1, 2, 2, 2], 10)
        rec = np.repeat(np.arange(6), 10)
        tmid = np.tile(np.arange(10) * 0.5 + 1.0, 6)
        folds = rooms.cv_folds(y, rec, pas, tmid)
        self.assertEqual(len(folds), 2)
        for test, train in folds:
            self.assertEqual(len(np.unique(pas[test])), 1)
            self.assertFalse(np.any(train & test))
            self.assertFalse(np.any(np.isin(rec[train], rec[test])))
            # every class learns from the same amount of data -> no bias against the tested one
            per_class = [len(np.unique(rec[train & (y == c)])) for c in range(3)]
            self.assertEqual(per_class, [1, 1, 1])

    def test_a_single_recording_never_trains_on_overlapping_windows(self):
        y = np.array([0] * 40 + [1] * 40 + [1] * 40)
        rec = np.repeat([0, 1, 2], 40)
        pas = np.repeat([1, 1, 2], 40)
        tmid = np.tile(np.arange(40) * rooms.HOP_S + 1.0, 3)
        folds = [f for f in rooms.cv_folds(y, rec, pas, tmid) if np.all(rec[f[0]] == 0)]
        self.assertEqual(len(folds), rooms.N_BLOCKS)
        for test, train in folds:
            same = train & (rec == 0)
            gaps = np.abs(tmid[same][:, None] - tmid[test][None, :])
            self.assertTrue(np.all(gaps >= rooms.WIN_S))   # windows overlap below WIN_S

    def test_zones_merge_transitively(self):
        S = np.ones((4, 4))
        S[0, 1] = S[1, 0] = 0.55
        S[1, 2] = S[2, 1] = 0.6
        self.assertEqual(rooms.merge_zones(S, 4), [[0, 1, 2], [3]])


# ──────────────────────────────────────────────────────────────────────────
class SeparableHouseTests(TrainedHouse, unittest.TestCase):
    def test_reports_measured_accuracy_for_every_room(self):
        rep = self.report
        self.assertEqual(rep["mode"], "knn")
        self.assertIn("held-out passes", rep["scheme"])
        self.assertNotIn("optimistic", rep["scheme"])
        self.assertGreater(rep["accuracy"], 0.95)
        self.assertAlmostEqual(rep["chance"], 1 / len(rooms.LABELS))
        self.assertEqual([c["label"] for c in rep["perClass"]], rooms.LABELS)
        self.assertTrue(all(c["scheme"] == "other-pass" for c in rep["perClass"]))
        C = np.array(rep["confusion"])
        self.assertEqual(C.shape, (len(rooms.LABELS),) * 2)
        self.assertEqual(C.sum(), rep["n"])
        self.assertTrue(all(len(z["members"]) == 1 and z["reliable"] for z in rep["zones"]))

    def test_live_output_names_the_room_with_a_hedged_confidence(self):
        out = live_on(self.dir, self.house, "deniz", self.t_end)
        self.assertEqual(out["state"], "live")
        self.assertEqual(out["mode"], "knn")
        self.assertEqual(out["top"], ["deniz"])
        self.assertFalse(out["merged"])
        self.assertTrue(out["reliable"])
        self.assertEqual(len(out["probs"]), len(rooms.LABELS))
        self.assertAlmostEqual(sum(out["probs"]), 1.0, places=2)
        eps = self.report["hedge"]
        self.assertLessEqual(out["conf"], 1 - eps + eps / len(rooms.LABELS) + 1e-3)

    def test_nobody_moving_is_a_learned_class(self):
        out = live_on(self.dir, self.house, rooms.NOBODY, self.t_end)
        self.assertEqual(out["top"], [rooms.NOBODY])
        out = live_on(self.dir, self.house, "kitchen", self.t_end)
        self.assertNotIn(rooms.NOBODY, out["top"])

    def test_output_is_smoothed_over_time(self):
        eng = rooms.RoomEngine(self.dir)
        buf = rooms.FrameBuffer()
        t, A, r = self.house.record("cem", secs=4.0, t0=self.t_end)
        feed(buf, t, A, r)
        self.assertEqual(eng.infer(buf, NODE, now=t[-1])["top"], ["cem"])
        t2, A2, r2 = self.house.record("ece", secs=6.0, t0=t[-1] + 0.01)
        tops, i = [], 0
        for now in t2[0] + rooms.WIN_S + 0.05 + np.arange(12) * rooms.INFER_EVERY_S:
            while i < len(t2) and t2[i] <= now:
                buf.push(t2[i], A2[i], r2[i]); i += 1
            tops.append(eng.infer(buf, NODE, now=now)["top"])
        self.assertEqual(tops[0], ["cem"])       # a full window of Ece doesn't flip it at once
        self.assertEqual(tops[-1], ["ece"])

    def test_nothing_is_shown_without_live_csi(self):
        buf = rooms.FrameBuffer()
        t, A, r = self.house.record("cem", secs=3.0, t0=self.t_end)
        feed(buf, t, A, r)
        out = self.engine.infer(buf, NODE, now=t[-1] + rooms.STALE_S + 0.5)
        self.assertEqual(out["state"], "no-signal")
        self.assertNotIn("top", out)

    def test_a_model_is_only_used_on_the_card_it_was_trained_on(self):
        buf = rooms.FrameBuffer()
        t, A, r = self.house.record("cem", secs=3.0, t0=self.t_end)
        feed(buf, t, A, r)
        out = self.engine.infer(buf, "other", now=t[-1])
        self.assertEqual(out["state"], "other-card")
        self.assertNotIn("top", out)


class InseparableHouseTests(TrainedHouse, unittest.TestCase):
    house_kw = dict(twins=[("bath1", "bath2")], blind=["bath3"])

    def test_rooms_the_data_cannot_separate_become_one_zone(self):
        zones = [sorted(z["members"]) for z in self.report["zones"]]
        self.assertIn(["bath1", "bath2"], zones)
        self.assertIn(sorted([rooms.NOBODY, "bath3"]), zones)
        self.assertTrue(any(p["a"] in ("bath1", "bath2") and p["b"] in ("bath1", "bath2")
                            for p in self.report["confusable"]))
        merged = [z for z in self.report["zones"] if len(z["members"]) > 1]
        self.assertTrue(all(not z["reliable"] for z in merged))

    def test_a_merged_room_is_never_reported_alone(self):
        for lab in ("bath1", "bath2"):
            out = live_on(self.dir, self.house, lab, self.t_end)
            self.assertEqual(sorted(out["top"]), ["bath1", "bath2"])
            self.assertTrue(out["merged"])
            self.assertFalse(out["reliable"])
            zp = out["zoneProbs"][out["zone"]]
            i1, i2 = rooms.LABELS.index("bath1"), rooms.LABELS.index("bath2")
            self.assertAlmostEqual(zp, out["probs"][i1] + out["probs"][i2], places=2)


class NoRoomSignalTests(TrainedHouse, unittest.TestCase):
    """Every 'room' recorded with nobody moving: an honest score is at chance."""
    @classmethod
    def setUpClass(cls):
        cls.house_kw = {}
        cls.dir = tempfile.mkdtemp(prefix="rooms-test-")
        cls.house = House(rooms.LABELS, seed=11)
        for lab in rooms.LABELS:
            cls.house.print[lab] = cls.house.print[rooms.NOBODY]
        cls.t_end = record_house(cls.dir, cls.house)
        cls.report = rooms.train(cls.dir, neural=False)
        cls.engine = rooms.RoomEngine(cls.dir)

    def test_no_signal_scores_at_chance_and_merges_everything(self):
        # neither inflated by the recording session nor dragged below chance by the folds
        chance = self.report["chance"]
        self.assertLess(self.report["accuracy"], 2.5 * chance)
        self.assertGreater(self.report["accuracy"], chance / 4)
        self.assertLessEqual(len(self.report["zones"]), 3)


class TrainingInputTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="rooms-test-")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_training_needs_every_class_including_nobody(self):
        house = House(rooms.LABELS, seed=2)
        t0 = 1.7e9
        for lab in rooms.LABELS[1:]:
            t, A, r = house.record(lab, t0=t0); t0 += 45
            rooms.save_recording(self.dir, lab, 1, NODE, t, A, r)
        with self.assertRaisesRegex(ValueError, "Nobody moving"):
            rooms.train(self.dir, neural=False)

    def test_a_crash_mid_replace_still_counts_one_recording_per_room_and_pass(self):
        house = House(rooms.LABELS, seed=2)
        t, A, r = house.record("cem", secs=6, t0=1.7e9)
        old = rooms.save_recording(self.dir, "cem", 1, NODE, t, A, r)
        keep = os.path.join(self.dir, "kept")
        shutil.copy(os.path.join(rooms.recordings_dir(self.dir), old), keep)
        t2, A2, r2 = house.record("cem", secs=6, t0=1.7e9 + 100)
        new = rooms.save_recording(self.dir, "cem", 1, NODE, t2, A2, r2)
        shutil.move(keep, os.path.join(rooms.recordings_dir(self.dir), old))
        recs = rooms.load_recordings(self.dir)
        self.assertEqual([x["file"] for x in recs], [new])
        cal = RoomCalibration(self.dir, rooms.RoomEngine(self.dir))
        self.assertEqual(cal.have["cem"][1]["file"], new)

    def test_untrained_engine_names_no_room(self):
        eng = rooms.RoomEngine(self.dir)
        buf = rooms.FrameBuffer()
        buf.push(time.time(), np.ones(64), -40)
        out = eng.infer(buf, NODE)
        self.assertEqual((out["mode"], out["state"]), ("untrained", "untrained"))
        self.assertNotIn("top", out)
        self.assertNotIn("probs", out)

    def test_single_pass_accuracy_is_labelled_optimistic(self):
        record_house(self.dir, House(rooms.LABELS, seed=4), passes=1)
        rep = rooms.train(self.dir, neural=False)
        self.assertIn("optimistic", rep["scheme"])
        self.assertTrue(all(c["scheme"] == "within-recording" for c in rep["perClass"]))

    def test_a_failing_neural_trial_leaves_a_working_knn(self):
        import types
        from unittest import mock
        import ai
        record_house(self.dir, House(rooms.LABELS, seed=8), passes=1, secs=12)
        fake = types.ModuleType("ai.rooms_neural")
        fake.T_STEPS = 8
        def boom(*a):
            raise RuntimeError("no GPU today")
        fake.train_and_compare = boom
        with mock.patch.dict(sys.modules, {"ai.rooms_neural": fake}), \
             mock.patch.object(ai, "rooms_neural", fake, create=True):
            rep = rooms.train(self.dir, neural=True)
        self.assertEqual(rep["mode"], "knn")
        self.assertIn("no GPU today", rep["neuralTrial"]["error"])
        eng = rooms.RoomEngine(self.dir)
        self.assertEqual((eng.mode, eng.report["neuralTrial"]["kept"]), ("knn", False))

    def test_retraining_removes_a_stale_neural_model(self):
        record_house(self.dir, House(rooms.LABELS, seed=5), passes=1, secs=12)
        for fn in (rooms.NEURAL_FILE, rooms.NEURAL_META):
            open(os.path.join(self.dir, fn), "w").close()
        rooms.train(self.dir, neural=False)
        self.assertFalse(os.path.exists(os.path.join(self.dir, rooms.NEURAL_FILE)))
        self.assertEqual(rooms.RoomEngine(self.dir).mode, "knn")


# ──────────────────────────────────────────────────────────────────────────
class FakeCard:
    def __init__(self, card=NODE):
        self.card = card
        self.room_buf = rooms.FrameBuffer()


class Clock:
    def __init__(self, t=1.8e9):
        self.t = t

    def __call__(self):
        return self.t


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="rooms-test-")
        self.clock = Clock()
        self.engine = rooms.RoomEngine(self.dir)
        self.cal = RoomCalibration(self.dir, self.engine, clock=self.clock)
        self.card = FakeCard()
        self.house = House(rooms.LABELS, seed=9)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def run_recording(self, label, pass_no, stream_s=None):
        """Drive the state machine like the DSP loop does (20 Hz), streaming CSI."""
        self.assertTrue(self.cal.start(label, pass_no))
        self.assertEqual(self.cal.status()["active"]["phase"], "countdown")
        stream_s = COUNTDOWN_S + REC_S + 1 if stream_s is None else stream_s
        t, A, r = self.house.record(label, secs=COUNTDOWN_S + REC_S + 2, t0=self.clock.t)
        i, end = 0, self.clock.t + COUNTDOWN_S + REC_S + GAP_S + 1
        while self.clock.t < end and (self.cal.active or i == 0):
            self.clock.t += 0.05
            while i < len(t) and t[i] <= self.clock.t and t[i] - t[0] < stream_s:
                self.card.room_buf.push(t[i], A[i], r[i]); i += 1
            self.cal.update(self.card)
        return self.cal.status()

    def test_guided_recording_is_saved_and_the_walk_resumes(self):
        st = self.cal.status()
        self.assertEqual(st["next"], dict(label=rooms.NOBODY, pass_no=1))
        self.assertEqual(st["total"], 2 * len(rooms.LABELS))
        st = self.run_recording(rooms.NOBODY, 1)
        self.assertIsNone(st["active"])
        self.assertIsNone(st["error"])
        self.assertAlmostEqual(st["have"][rooms.NOBODY]["1"]["secs"], REC_S, delta=1.0)
        self.assertEqual(st["next"], dict(label="bedroom", pass_no=1))
        again = RoomCalibration(self.dir, self.engine, clock=self.clock).status()  # restart
        self.assertEqual(again["have"], st["have"])
        self.assertEqual(again["next"], st["next"])

    def test_one_room_can_be_recorded_again_without_touching_the_rest(self):
        self.run_recording(rooms.NOBODY, 1)
        self.run_recording("cem", 1)
        before = sorted(os.listdir(rooms.recordings_dir(self.dir)))
        self.run_recording("cem", 1)
        after = sorted(os.listdir(rooms.recordings_dir(self.dir)))
        self.assertEqual(len(after), 2)
        self.assertIn([f for f in before if f.startswith("nobody")][0], after)
        self.assertNotEqual([f for f in before if f.startswith("cem")],
                            [f for f in after if f.startswith("cem")])

    def test_a_recording_stops_and_saves_nothing_when_csi_stops(self):
        st = self.run_recording("ece", 1, stream_s=COUNTDOWN_S + 4)
        self.assertIn("CSI stopped", st["error"])
        self.assertEqual(st["have"], {})

    def test_bad_requests_are_refused(self):
        self.assertFalse(self.cal.start("garage", 1))
        self.assertFalse(self.cal.start("cem", 3))
        self.assertTrue(self.cal.start("cem", 1))
        self.assertFalse(self.cal.start("ece", 1))           # one at a time
        self.cal.handle(dict(action="cancel"))
        self.assertIsNone(self.cal.status()["active"])

    def test_training_from_the_page_switches_the_engine_on(self):
        record_house(self.dir, self.house, passes=2, secs=12)
        self.cal._scan()
        self.assertTrue(self.cal.status()["canTrain"])
        self.cal.handle(dict(action="train"))
        deadline = time.time() + 60
        while self.cal.status()["train"]["state"] == "running" and time.time() < deadline:
            time.sleep(0.05)
        st = self.cal.status()
        self.assertEqual(st["train"]["state"], "done", st["train"]["msg"])
        self.assertIn("accuracy", st["train"]["msg"])
        self.assertEqual(self.engine.mode, "knn")
        self.assertFalse(st["modelStale"])
        self.cal.handle(dict(action="discard"))
        self.assertEqual(self.cal.status()["done"], 0)
        self.assertEqual(self.engine.mode, "untrained")

    def test_training_without_every_room_explains_what_is_missing(self):
        self.run_recording(rooms.NOBODY, 1)
        self.cal.train()
        deadline = time.time() + 30
        while self.cal.status()["train"]["state"] == "running" and time.time() < deadline:
            time.sleep(0.05)
        st = self.cal.status()["train"]
        self.assertEqual(st["state"], "error")
        self.assertIn("Bedroom", st["msg"])


# ──────────────────────────────────────────────────────────────────────────
class ServerWiringTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="rooms-test-")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_card_frames_reach_the_room_classifier_and_page_commands_work(self):
        import sgpcard
        sgpcard.ROOMS = rooms.RoomEngine(self.dir)
        sgpcard.ROOM_CAL = RoomCalibration(self.dir, sgpcard.ROOMS)
        e = sgpcard.NodeEngine("A")
        e.card = "192.168.4.1"
        iq = np.random.default_rng(0).integers(5, 30, 128).astype(np.int8).tobytes()
        e.feed(0, -40, -90, iq, (0, 0, 0, 0), 0)
        self.assertEqual(e.room_buf.seq, 1)
        st = sgpcard.room_state([e])
        self.assertEqual((st["live"]["mode"], st["live"]["state"]), ("untrained", "untrained"))
        self.assertIsNone(st["report"])
        sgpcard.apply_client_msg(dict(type="rooms", action="record", label="nobody", **{"pass": 1}))
        self.assertEqual(sgpcard.ROOM_CAL.status()["active"]["label"], "nobody")
        self.assertEqual(sgpcard.room_state([])["live"]["state"], "no-card")


class _PlanParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rooms, self.labels = [], []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "polygon" and "data-room" in a:
            self.rooms.append(a["data-room"])
        elif tag == "g" and "data-label" in a:
            self.labels.append(a["data-label"])


class PlanContractTests(unittest.TestCase):
    def test_the_plan_draws_each_classifier_room_once(self):
        p = _PlanParser()
        with open(os.path.join(PC, "web", "rooms.html")) as f:
            p.feed(f.read())
        ids = sorted(r for r, _ in rooms.ROOMS)
        self.assertEqual(sorted(p.rooms), ids)
        self.assertEqual(sorted(p.labels), ids)


class ServedAddressTests(unittest.TestCase):
    def test_pages_are_opened_on_the_address_the_server_listens_on(self):
        import sgpcard
        self.assertEqual(sgpcard.page_hosts("127.0.0.1"), ("127.0.0.1", None))
        self.assertEqual(sgpcard.page_hosts("192.168.1.5"), ("192.168.1.5", "192.168.1.5"))
        with mock.patch.object(sgpcard, "my_lan_ip", return_value="10.0.0.7"):
            self.assertEqual(sgpcard.page_hosts("0.0.0.0"), ("127.0.0.1", "10.0.0.7"))


@unittest.skipUnless(importlib.util.find_spec("websockets"), "websockets not installed")
class WebsocketOriginTests(unittest.TestCase):
    def setUp(self):
        import sgpcard
        self.sg = sgpcard
        self.dir = tempfile.mkdtemp(prefix="rooms-test-")
        record_house(self.dir, House(rooms.LABELS, seed=2), passes=1, secs=6)
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self.saved = (sgpcard.HOST, sgpcard.PORT, dict(sgpcard.HOST_INFO), sgpcard.ROOMS, sgpcard.ROOM_CAL)
        sgpcard.HOST, sgpcard.PORT = "127.0.0.1", port - 1
        sgpcard.HOST_INFO.clear()
        sgpcard.ROOMS = rooms.RoomEngine(self.dir)
        sgpcard.ROOM_CAL = RoomCalibration(self.dir, sgpcard.ROOMS)

    def tearDown(self):
        sg = self.sg
        sg.HOST, sg.PORT, info, sg.ROOMS, sg.ROOM_CAL = self.saved
        sg.HOST_INFO.clear(); sg.HOST_INFO.update(info)
        shutil.rmtree(self.dir, ignore_errors=True)

    def _discard_from(self, origin):
        import websockets
        sg = self.sg
        url = "ws://127.0.0.1:%d" % (sg.PORT + 1)

        async def go():
            async with sg.ws_server():
                try:
                    async with websockets.connect(url, origin=origin) as ws:
                        await ws.recv()
                        await ws.send(json.dumps(dict(type="rooms", action="discard")))
                        await asyncio.sleep(0.3)
                    return True
                except websockets.exceptions.InvalidHandshake:
                    return False
        return asyncio.run(go())

    def _handshake(self, host, origin):
        """HTTP status of a websocket handshake carrying these Host / Origin headers
        (a tuple sends one header per value)."""
        hosts = (host,) if isinstance(host, str) else host
        origins = (origin,) if isinstance(origin, str) else origin or ()
        sg = self.sg

        async def go():
            async with sg.ws_server():
                r, w = await asyncio.open_connection("127.0.0.1", sg.PORT + 1)
                w.write(("GET / HTTP/1.1\r\n%sUpgrade: websocket\r\n"
                         "Connection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
                         "Sec-WebSocket-Version: 13\r\n%s\r\n"
                         % ("".join("Host: %s\r\n" % h for h in hosts), "".join("Origin: %s\r\n" % o for o in origins))).encode())
                await w.drain()
                status = int((await r.readline()).split()[1])
                w.close()
                return status
        return asyncio.run(go())

    def test_the_page_connects_however_this_computer_was_reached(self):
        p = self.sg.PORT
        for name in ("127.0.0.1", "localhost", "192.168.4.2", "10.0.0.7", "mac-mini.local",
                     socket.gethostname().lower(), "[::1]"):
            with self.subTest(name=name):
                self.assertEqual(self._handshake("%s:%d" % (name, p + 1), "http://%s:%d" % (name, p)), 101)
        self.assertEqual(self._handshake("127.0.0.1:%d" % (p + 1), None), 101)

    def test_pages_from_elsewhere_are_refused(self):
        p = self.sg.PORT
        for host, origin in (("127.0.0.1", "http://evil.example"),
                             ("evil.example", "http://evil.example:%d" % p),
                             ("10.0.0.7", "http://192.168.4.2:%d" % p),
                             ("10.0.0.7", "http://10.0.0.7:8000"),
                             ("10.0.0.7", "https://10.0.0.7:%d" % p)):
            with self.subTest(host=host, origin=origin):
                self.assertEqual(self._handshake("%s:%d" % (host, p + 1), origin), 403)

    def test_a_handshake_with_two_host_or_origin_headers_is_refused(self):
        p = self.sg.PORT
        host, page = "127.0.0.1:%d" % (p + 1), "http://127.0.0.1:%d" % p
        for hosts, origins in (((host, "evil.example:%d" % (p + 1)), page),
                               ((host,) * 2, page),
                               (host, (page, "http://evil.example")),
                               (host, (page,) * 2)):
            with self.subTest(hosts=hosts, origins=origins):
                self.assertEqual(self._handshake(hosts, origins), 403)

    def test_a_page_from_another_site_cannot_wipe_the_recordings(self):
        self.assertFalse(self._discard_from("http://evil.example"))
        self.assertTrue(rooms.load_recordings(self.dir))

    def test_the_house_map_page_can(self):
        self.assertTrue(self._discard_from("http://127.0.0.1:%d" % self.sg.PORT))
        self.assertEqual(rooms.load_recordings(self.dir), [])


@unittest.skipUnless(rooms._torch_available(), "torch not installed")
class NeuralUpgradeTests(unittest.TestCase):
    def test_neural_model_is_tried_and_only_kept_if_it_wins(self):
        from ai import rooms_neural
        d = tempfile.mkdtemp(prefix="rooms-test-")
        try:
            record_house(d, House(rooms.LABELS, seed=6), passes=2, secs=12)
            old = rooms_neural.EPOCHS
            rooms_neural.EPOCHS = 3
            try:
                rep = rooms.train(d)
            finally:
                rooms_neural.EPOCHS = old
            trial = rep["neuralTrial"]
            self.assertIn("neuralAcc", trial)
            eng = rooms.RoomEngine(d)
            self.assertEqual(eng.mode, "neural" if trial["kept"] else "knn")
            self.assertEqual(eng.report["mode"], eng.mode)
            self.assertEqual(os.path.exists(os.path.join(d, rooms.NEURAL_FILE)), trial["kept"])
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_engine_switches_to_a_neural_model_that_won(self):
        from unittest import mock
        from ai import rooms_neural
        d = tempfile.mkdtemp(prefix="rooms-test-")
        rng = np.random.default_rng(0)
        guess = lambda self, X: rng.random((len(np.atleast_2d(X)), self.n_cls))
        try:
            house = House(rooms.LABELS, seed=6)
            t_end = record_house(d, house, passes=2, secs=12)
            with mock.patch.object(rooms_neural, "EPOCHS", 6), \
                 mock.patch.object(rooms.LocalMeanKNN, "distances", guess):   # a k-NN that guesses
                rep = rooms.train(d)
            self.assertTrue(rep["neuralTrial"]["kept"], rep["neuralTrial"])
            self.assertEqual(rep["mode"], "neural")
            eng = rooms.RoomEngine(d)
            self.assertEqual((eng.mode, eng.report["mode"]), ("neural", "neural"))
            out = live_on(d, house, "cem", t_end)
            self.assertEqual((out["state"], out["mode"]), ("live", "neural"))
            self.assertAlmostEqual(sum(out["probs"]), 1.0, places=2)
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
