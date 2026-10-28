"""Tests for the House map room classifier, its guided calibration and the page.

Run from the repository root:  python -m unittest discover -s pc/tests
(numpy only; the neural test runs when torch is installed)."""
import os, re, shutil, sys, tempfile, time, unittest
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


class PageTests(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(PC, "web", "rooms.html")) as f:
            self.html = f.read()

    def test_the_plan_traces_every_room_once(self):
        ids = re.findall(r'data-room="(\w+)"', self.html)
        self.assertEqual(sorted(ids), sorted(r for r, _ in rooms.ROOMS))
        labels = re.findall(r'data-label="(\w+)"', self.html)
        self.assertEqual(sorted(labels), sorted(r for r, _ in rooms.ROOMS))
        self.assertIn('href="floorplan.png"', self.html)
        self.assertTrue(os.path.exists(os.path.join(PC, "web", "floorplan.png")))

    def test_the_page_cannot_invent_occupancy(self):
        # no demo / placeholder path: everything drawn comes from the server
        self.assertNotIn("Math.random", self.html)
        self.assertNotRegex(self.html.lower(), r"demo")

    def test_the_3d_view_links_to_the_house_map(self):
        with open(os.path.join(PC, "web", "index.html")) as f:
            self.assertIn("rooms.html", f.read())


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
