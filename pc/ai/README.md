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

## Credits (adapted model recipe, MIT)
- **SenseFi / WiFi-CSI-Sensing-Benchmark** — Yang et al., *Patterns* 2023
  (github.com/xyanchen/WiFi-CSI-Sensing-Benchmark, MIT). Reference model zoo in
  `vendor/SenseFi`.
- **ArielM1904/Wifi-CSI** (MIT) CNN/LSTM HAR pipeline, reference in
  `vendor/ArielM1904-WifiCSI`.
Our `model.py` (compact CNN+GRU) is adapted to the SGP Card Mini's CSI format.
