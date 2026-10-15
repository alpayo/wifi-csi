"""
SGP CSI WIFI — neural CSI classifier.

A compact CNN + bidirectional-GRU that takes a short window of our CSI amplitude
(time x subcarriers) and classifies the person's state. The architecture follows
the proven recipe from the SenseFi / WiFi-CSI-Sensing-Benchmark work
(xyanchen/WiFi-CSI-Sensing-Benchmark, MIT) — 1D conv feature extractor over the
subcarriers + a recurrent temporal head — adapted to OUR input shape (any number
of subcarriers, thanks to the adaptive pooling) so it runs on the SGP Card Mini.

This is CLASSIFICATION (empty / still / walking / fast / fall), not metric (x,y,z)
localization. It makes presence/activity far more robust once trained on data
recorded with your own card (see record.py + train.py).
"""
import torch
import torch.nn as nn

CLASSES = ["empty", "still", "walking", "fast", "fall"]

class CSINet(nn.Module):
    def __init__(self, n_cls=len(CLASSES), pool=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.AdaptiveAvgPool1d(pool),          # -> fixed size, any #subcarriers
        )
        self.feat = 32 * pool
        self.gru = nn.GRU(self.feat, 64, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, n_cls))

    def forward(self, x):            # x: (B, T, S) CSI amplitude window
        B, T, S = x.shape
        x = x.reshape(B * T, 1, S)
        x = self.conv(x).reshape(B, T, -1)
        o, _ = self.gru(x)
        return self.head(o[:, -1])   # last time step
