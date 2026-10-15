import os, glob, yaml
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split

# ================= CONFIG =================
with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

DATA_ROOT = cfg["data_root"]
SAVE_PATH = cfg["processed_path"]

FS = 60.0
BP_LOW, BP_HIGH = 0.1, 0.5


WIN_LEN = 300
STRIDE = 100

CLASS_NAMES = ["LOS_AIR", "LOS_BREATH", "NLOS_AIR", "NLOS_BREATH"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASS_NAMES)}

# ================= HAMPEL FILTER =================
def hampel_filter(X, k=7, t0=3):
    Xf = X.copy()
    for i in range(X.shape[1]):
        series = X[:, i]
        median = pd.Series(series).rolling(k, center=True).median()
        diff = np.abs(series - median)
        mad = diff.rolling(k, center=True).median()
        threshold = t0 * 1.4826 * mad
        outliers = diff > threshold
        series[outliers] = median[outliers]
        Xf[:, i] = series
    return Xf

# ================= LOAD =================
def load_csv(path):
    df = pd.read_csv(path)

    print(f"\n📄 Archive: {path}")
    print("Shape original:", df.shape)

    ts = df["timestamp"].values.astype(np.float64)

    # Detectar columnas CSI dinámicamente
    csi_cols_detected = [c for c in df.columns if c.startswith("csi_")]

    if len(csi_cols_detected) != 274:
        print(f"⚠️ Skipping file (incorrect dim) {len(csi_cols_detected)}): {path}")
        return None, None

    X = df[csi_cols_detected].fillna(0).values.astype(np.float32)

    print("Shape CSI:", X.shape)

    return ts - ts[0], X

# ================= RESAMPLE =================
def resample(t, X):
    t_new = np.arange(t[0], t[-1], 1.0 / FS)
    f = interp1d(t, X, axis=0, fill_value="extrapolate")
    return f(t_new)

# ================= BANDPASS =================
def bandpass(X):
    nyq = 0.5 * FS
    b, a = butter(4, [BP_LOW / nyq, BP_HIGH / nyq], btype="band")
    return filtfilt(b, a, X, axis=0)

# ================= WINDOW =================
def create_windows(X):
    if X.shape[0] < WIN_LEN:
        return np.array([])

    windows = []

    for i in range(0, X.shape[0] - WIN_LEN, STRIDE):
        w = X[i:i + WIN_LEN]

        # Normalización Z-score
        w = (w - w.mean()) / (w.std() + 1e-8)

        windows.append(w)

    if len(windows) == 0:
        return np.array([])

    return np.stack(windows)

# ================= MAIN =================
os.makedirs(SAVE_PATH, exist_ok=True)

X_all, y_all, groups = [], [], []
group_id = 0

for cname in CLASS_NAMES:
    files = glob.glob(os.path.join(DATA_ROOT, cname, "*.csv"))

    print(f"\n📂 Processing class: {cname} | Files: {len(files)}")

    for fpath in files:
        try:
            t, X = load_csv(fpath)

            if X is None:
                continue

            X = resample(t, X)
            X = hampel_filter(X)
            X = bandpass(X)

            W = create_windows(X)

            if W.size == 0:
                print(f"⚠️ No windows: {fpath}")
                continue

            y = np.full(len(W), CLASS_TO_ID[cname])
            g = np.full(len(W), group_id)

            X_all.append(W)
            y_all.append(y)
            groups.append(g)

            group_id += 1

        except Exception as e:
            print(f"❌ Error in {fpath}: {e}")

# ================= VALIDACIÓN =================
if len(X_all) == 0:
    raise ValueError("❌ No windows were generated.")

print("\n📊 Shapes individuales:")
for i, arr in enumerate(X_all[:5]):
    print(f"Ejemplo {i}: {arr.shape}")

X_all = np.concatenate(X_all)
y_all = np.concatenate(y_all)
groups = np.concatenate(groups)

print("\n✅ Total dataset:", X_all.shape)

# ================= SPLIT =================
print("\n🔀 Generating train/test split...")

X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
    X_all, y_all, groups,
    test_size=0.2,
    stratify=y_all,
    random_state=42
)

# ================= SAVE =================
np.save(os.path.join(SAVE_PATH, "X.npy"), X_all)
np.save(os.path.join(SAVE_PATH, "y.npy"), y_all)
np.save(os.path.join(SAVE_PATH, "groups.npy"), groups)

np.save(os.path.join(SAVE_PATH, "X_train.npy"), X_train)
np.save(os.path.join(SAVE_PATH, "y_train.npy"), y_train)
np.save(os.path.join(SAVE_PATH, "X_test.npy"), X_test)
np.save(os.path.join(SAVE_PATH, "y_test.npy"), y_test)

print("\n✅ Train:", X_train.shape)
print("✅ Test:", X_test.shape)
