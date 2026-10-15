import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score
)

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="nlos",
                    choices=["nlos", "multiclass"],
                    help="Modo de evaluación")
args = parser.parse_args()


X = np.load("data/processed/X_test.npy")
y = np.load("data/processed/y_test.npy")

print("\n📊 Dataset cargado:")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Clases únicas:", np.unique(y))



if args.mode == "nlos":
    print("\n🔍 Modo NLOS (binario)")

    unique_classes = np.unique(y)

    
    if set(unique_classes) == {0, 1}:
        print("✅ Dataset ya está en formato binario (0=AIR, 1=BREATH)")
        class_names = ["AIR", "BREATH"]

  
    elif set(unique_classes).issubset({0,1,2,3}):
        print("⚠️ Dataset in multiclass format → applying NLOS filter")

        mask = (y == 2) | (y == 3)
        X = X[mask]
        y = y[mask]

        if len(X) == 0:
            raise ValueError("❌ No NLOS data after filtering.")

        # convertir a binario
        y = (y == 3).astype(int)

        class_names = ["AIR", "BREATH"]

    else:
        raise ValueError(f"❌ Unknown classes: {unique_classes}")

else:
    print("\n🔍 MULTICLASS Mode")

    if len(np.unique(y)) <= 2:
        raise ValueError("❌ The dataset is binary; you cannot use multiclass mode.")

    class_names = ["LOS_AIR", "LOS_BREATH", "NLOS_AIR", "NLOS_BREATH"]


model = tf.keras.models.load_model("models/best_model.keras")


expected_shape = model.input_shape[1:]

if X.shape[1:] != expected_shape:
    raise ValueError(
        f"❌ Shape mismatch: expect model {expected_shape}, but received {X.shape[1:]}"
    )


print("\n🤖 Generating predictions...")
y_prob = model.predict(X, verbose=0)

if y_prob.shape[0] == 0:
    raise ValueError("❌ Prediction failed: empty dataset.")

y_pred = np.argmax(y_prob, axis=1)


print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y, y_pred, zero_division=0))

os.makedirs("results", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)


cm = confusion_matrix(y, y_pred)

# Normalizada (paper-level)
cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

plt.figure()
sns.heatmap(cm_norm, annot=True, fmt=".2f",
            xticklabels=class_names,
            yticklabels=class_names)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix NORMALIZED ({args.mode.upper()})")
plt.savefig(f"results/figures/confusion_matrix_{args.mode}.png")
plt.close()


if args.mode == "nlos":
    print("\n📈 Calculating ROC (binary)...")

    try:
        fpr, tpr, _ = roc_curve(y, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], '--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (NLOS)")
        plt.legend()
        plt.savefig("results/figures/roc_curve_nlos.png")
        plt.close()

        print(f"AUC: {roc_auc:.4f}")

    except Exception as e:
        print(f"⚠️ Error calculating ROC: {e}")

else:
    print("\n📈 Calculating multiclass AUC (OvR)...")

    try:
        roc_auc = roc_auc_score(y, y_prob, multi_class="ovr")
        print(f"AUC (OvR): {roc_auc:.4f}")
    except Exception as e:
        print(f"⚠️ Could not calculate multiclass AUC: {e}")


import pandas as pd

results = pd.DataFrame({
    "mode": [args.mode],
    "samples": [len(X)],
})

results.to_csv(f"results/metrics_{args.mode}.csv", index=False)

print("\n📁 Results saved in /results/")
print("📊 Figures saved in /results/figures/")
