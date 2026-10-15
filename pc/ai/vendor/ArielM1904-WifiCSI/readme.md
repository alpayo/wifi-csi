# 📡 WiFi CSI-Based Human Activity Recognition (NLoS Focus)

This repository contains the full implementation of a deep learning pipeline for **Human Activity Recognition (HAR)** using WiFi **Channel State Information (CSI)**, with a primary focus on **Non-Line-of-Sight (NLoS) respiration detection**.

The system is based on a **CNN-LSTM architecture** and provides a fully reproducible workflow from raw data preprocessing to model evaluation and visualization.

---

## 🚀 Project Overview

This work explores the feasibility of detecting human respiration through obstacles using WiFi CSI signals.

The proposed approach combines signal processing techniques with deep learning to extract meaningful patterns from CSI data in challenging propagation conditions.

---

## 🔥 Main Contributions

- 📶 CSI preprocessing pipeline:
  - Resampling
  - Hampel filtering (outlier removal)
  - Bandpass filtering (respiration frequency range)

- 🧠 Deep learning architecture:
  - 1D Convolutional Neural Network (CNN)
  - Long Short-Term Memory (LSTM)

- 🧪 Robust data splitting:
  - Group-based split to avoid data leakage

- 🎯 Focused experiment:
  - NLoS respiration detection (binary classification)

- ⚠️ Additional experiment:
  - Multiclass classification (LOS/NLOS), included for analysis

---

## 🧠 Experiments

### ✅ Main Experiment (Paper Contribution)

**NLoS Scenario (Binary Classification)**

Classes:
- `NLOS_AIR` → No breathing  
- `NLOS_BREATH` → Breathing  

Reformulated as:
- `0 = AIR`  
- `1 = BREATH`  

✔ This is the **core contribution of the paper**  
✔ Achieves high performance in challenging NLoS conditions  

---

### ⚠️ Secondary Experiment (Exploratory)

**Multiclass Classification**

Classes:
- `LOS_AIR`
- `LOS_BREATH`
- `NLOS_AIR`
- `NLOS_BREATH`

Limitations observed:
- Environmental variability  
- Channel instability  
- Climatic conditions affecting CSI signals  

👉 This experiment is included for completeness but is **not the main contribution**.

---

## 📂 Project Structure

```
Proyecto/
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── preprocessing/
│   └── preprocess.py
├── training/
│   └── train.py
├── evaluation/
│   └── evaluate.py
├── models/
│   ├── cnn_lstm.py
│   └── best_model.keras
├── results/
│   ├── metrics.csv
│   └── figures/
├── run_all.sh
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

```bash
git clone https://github.com/ArielM1904/Wifi-CSI
cd Wifi-CSI

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

---

## ▶️ Usage

### 🔹 Run Full Pipeline

```bash
bash run_all.sh
```

This executes:
1. Preprocessing  
2. Training (NLoS)  
3. Evaluation  

---

### 🔹 Step-by-Step Execution

#### 1. Preprocessing

```bash
python preprocessing/preprocess.py
```

Generates:
- `data/processed/X.npy`
- `data/processed/y.npy`
- `data/processed/groups.npy`

---

#### 2. Training (Main Experiment - NLoS)

```bash
python training/train.py --mode nlos
```

Outputs:
- `models/best_model.keras`
- `results/metrics.csv`
- Test split saved for reproducibility

---

#### 3. Evaluation

```bash
python evaluation/evaluate.py --mode nlos
```

Generates:
- Confusion matrix  
- ROC curve  
- Classification report  

Saved in:
```
results/figures/
```

---

## 📊 Results

Typical performance in NLoS scenario:

- Accuracy: ~0.97 – 0.98  
- F1-score: ~0.98  
- AUC: ~0.987  

📈 These results demonstrate strong capability for respiration detection through obstacles using WiFi CSI.

---

## ⚠️ Notes

- GPU is **not required** (CPU execution supported)  
- CUDA warnings can be ignored  
- Group-based split prevents data leakage  
- Multiclass results may be unstable due to environmental factors  

---

## 📜 License

MIT License

---

## 👨‍💻 Authors

Ariel Mora
Diego Andrade 
Deyvi Totoy
