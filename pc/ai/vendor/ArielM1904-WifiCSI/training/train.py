import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from models.cnn_lstm import build_model

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

X = np.load("data/processed/X.npy")
y = np.load("data/processed/y.npy")
groups = np.load("data/processed/groups.npy")

# 2 = NLOS_AIR, 3 = NLOS_BREATH
mask = (y == 2) | (y == 3)
X, y, groups = X[mask], y[mask], groups[mask]

# binario: 0 = AIR, 1 = BREATH
y = (y == 3).astype(int)

print(f"\n📊 Dataset NLOS: {X.shape}")

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
groups_train = groups[train_idx]


gss_val = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx2, val_idx = next(gss_val.split(X_train, y_train, groups_train))

X_val = X_train[val_idx]
y_val = y_train[val_idx]

X_train = X_train[train_idx2]
y_train = y_train[train_idx2]

print(f"✅ Train: {X_train.shape}")
print(f"✅ Val:   {X_val.shape}")
print(f"✅ Test:  {X_test.shape}")


model = build_model((X_train.shape[1], X_train.shape[2]), num_classes=2)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-5
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "models/best_model.keras",
        monitor="val_loss",
        save_best_only=True
    )
]


history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=60,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)


pd.DataFrame(history.history).to_csv("results/history.csv", index=False)


y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

acc = np.mean(y_pred == y_test)
f1 = f1_score(y_test, y_pred, average="binary")
auc = roc_auc_score(y_test, y_pred_prob[:, 1])

print("\n=== TEST RESULTS ===")
print(f"Accuracy: {acc:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"AUC: {auc:.4f}")


report = classification_report(y_test, y_pred, output_dict=True)
pd.DataFrame(report).transpose().to_csv("results/classification_report.csv")


results = pd.DataFrame({
    "accuracy": [acc],
    "f1_score": [f1],
    "auc": [auc]
})
results.to_csv("results/metrics.csv", index=False)

np.save("data/processed/X_test.npy", X_test)
np.save("data/processed/y_test.npy", y_test)

print("\n✅ Test split saved successfully")
print("📁 Results saved in /results/")
print("💾 Template saved in /models/")
