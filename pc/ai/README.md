# AI analysis (neural CSI classifier)

An optional deep-learning stage that classifies the person's state from the raw
CSI window: **empty / still / walking / fast / fall**.

- Works **immediately** in RULES mode (no training, no torch needed) using the
  presence/motion/breathing features.
- Upgrades to **NEURAL** mode once you train `model.pt` on data recorded with
  **your own** card — far more robust/accurate.

## Make it neural (train on your hardware)
```bash
pip install -r ../requirements-ai.txt        # torch
# connect PC to the card's WiFi, then record each behaviour (act it out):
python record.py --label empty   --secs 40
python record.py --label still   --secs 40
python record.py --label walking --secs 60
python record.py --label fast    --secs 40
python record.py --label fall    --secs 30     # safely
python train.py                               # writes model.pt
```
Restart the program → the Sensor panel's "Activity · AI" shows **neural**.

## Honest scope
This is activity/state **classification** (as in the research below), not metric
(x,y,z) localization — exact position still needs multiple anchor cards. The AI
makes presence/activity much more robust and is the base for adding your own
labels (gestures, specific events).

# Room classifier (House map)

`rooms.py` answers a different question: **which room is someone moving in?**
Its labels are the rooms of the house on `web/floorplan.png` plus **nobody moving**, and the House map page (`web/rooms.html`, also under **House map** in the 3D view's menu) shows the answer on the floor plan.

- Works with **numpy only** (local-mean k-nearest-neighbour over each subcarrier's gain-normalised median and spread), so a plain `run.command` install can use it.
- Upgrades to a **neural** room CSINet (`rooms_neural.py`) by itself when torch is installed and that model beat the k-NN on the same held-out recordings.
- Knows **nothing** until you teach it: untrained, the page says so and shows only the whole-home readings (presence, motion, breathing, distance), which need no training.

## Teach it your house (from the page)
1. Start the program; to walk around with a phone, start it with `./run.command --host 0.0.0.0` and open the printed address on the phone.
2. Open **House map → Calibrate**. The page names a room, counts down 5 s, and records 30 s while you walk around that room waving your arms; then you go to the next one.
3. Every room and "nobody moving" is recorded **twice** (pass 1 through the whole house, then pass 2): 22 recordings, about 16 minutes. Everyone else stays still meanwhile.
4. Press **Train**. It takes seconds and reports the measured accuracy.

Recordings are saved as they finish (`ai/data/rooms/`, or `--room-data DIR`), so a restart loses nothing; press any cell of the table to redo just that recording.
`python ai/train_rooms.py` trains the same way from the command line.

## Honest scope
One card and one computer are a single radio link, which cannot geometrically locate anyone.
Rooms are recognised as learned fingerprints, and how well they separate is measured, not promised:
- Accuracy is scored by holding out whole recordings (the other pass of each room), so it can't come from recognising the recording session.
- Rooms the data can't tell apart are **merged** into one hatched zone on the map; rooms with low accuracy get a dashed outline. Neither is ever shown as a sure answer.
- The confidence shown is calibrated on the held-out recordings and never claims more certainty than the number of recordings supports.
- With one card (one radio link, until more cards are added) it names the **one strongest source of movement**, not each room's occupancy on its own: a second person moving elsewhere won't show.
- It senses **movement**: someone sitting or lying still reads as "nobody moving".
- Doors, furniture or a moved card change the fingerprints: re-record the affected rooms and train again.

## Credits (adapted model recipe, MIT)
- **SenseFi / WiFi-CSI-Sensing-Benchmark** — Yang et al., *Patterns* 2023
  (github.com/xyanchen/WiFi-CSI-Sensing-Benchmark, MIT). Reference model zoo in
  `vendor/SenseFi`.
- **ArielM1904/Wifi-CSI** (MIT) CNN/LSTM HAR pipeline, reference in
  `vendor/ArielM1904-WifiCSI`.
Our `model.py` (compact CNN+GRU) is adapted to the SGP Card Mini's CSI format.
