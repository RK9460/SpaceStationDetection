# 🚀 Space Station Safety Detection — YOLOv8

A lightweight and portable **YOLOv8-based object detection pipeline** to detect critical safety equipment in space-station–style or industrial environments.

---

## 👥 Team

- **Team Name:** Tech Syndicate  
- **Members:**  
  - Rahul Kala  
  - Ravinder Singh

---

## 📁 Project Structure

```
GenIgnite/
├─ README.md
├─ Hackathon2_scripts/                 # Core project scripts
│  ├─ train.py                         # Adaptive YOLOv8 training
│  ├─ predict.py                       # Single/batch inference
│  ├─ generate_report.py               # CSV + bar chart reporting
│  ├─ visualize.py                     # (Optional visualization tools)
│  ├─ yolo_params.yaml                 # Dataset config (relative paths)
│  └─ ENV_SETUP/
│     └─ classes.txt                   # Class names (1 per line)
│
└─ Hackathon2_test/                    # 📂 Place dataset here (not in repo)
   ├─ train/
   │  ├─ images/
   │  └─ labels/
   ├─ val/
   │  ├─ images/
   │  └─ labels/
   └─ test/
      ├─ images/
      └─ labels/
```

> ⚠️ **Note:** The dataset folder `Hackathon2_test` is **not committed**. Place it manually as shown above.

---

## 🧰 Requirements

- Python **3.10+**  
- GPU optional (**CUDA recommended**)

```bash
pip install -U ultralytics matplotlib requests
```

👉 For GPU acceleration, install CUDA-enabled PyTorch from:  
https://pytorch.org

---

## 🧠 Dataset Configuration

**File:** `Hackathon2_scripts/yolo_params.yaml`

```yaml
path: ../Hackathon2_test
train: train/images
val:   val/images
test:  test/images

nc: 7
names: ['OxygenTank','NitrogenTank','FirstAidBox','FireAlarm','SafetySwitchPanel','EmergencyPhone','FireExtinguisher']
```

**Folder format:**

```
Hackathon2_test/
├─ train/
│  ├─ images/
│  └─ labels/
├─ val/
│  ├─ images/
│  └─ labels/
└─ test/
   ├─ images/
   └─ labels/
```

**Label format (YOLO):**
```
<class_id> <x_center> <y_center> <width> <height>
```
_All coordinates are normalized to `[0, 1]`._

---

## 🧪 Training

```bash
python Hackathon2_scripts/train.py   --yaml Hackathon2_scripts/yolo_params.yaml   --weights weights/yolov8n.pt   --results runs/detect
```

**Features:**
- ✅ Auto-downloads weights if missing  
- ✅ Adapts image size & batch size for low-VRAM GPUs  
- ✅ Saves results in `runs/detect/train_auto/`

---

## 🔍 Prediction (Inference)

### 📁 Predict on entire test set

```bash
python Hackathon2_scripts/predict.py   --weights runs/detect/train_auto/weights/best.pt   --source Hackathon2_test/test/images   --out-images predictions/images   --out-labels predictions/labels   --conf 0.25 --imgsz 640 --iou 0.45 --augment
```

### 🖼️ Predict on single image

```bash
python Hackathon2_scripts/predict.py   --weights runs/detect/train_auto/weights/best.pt   --source Hackathon2_test/test/images/example.png
```

**Outputs:**
- Annotated images → `predictions/images`  
- YOLO-format labels (with confidence) → `predictions/labels`

---

## 📊 Report Generation

Create automatic CSVs, a text summary, and a bar chart of class counts.

**Using YAML:**

```bash
python Hackathon2_scripts/generate_report.py   --pred predictions   --yaml Hackathon2_scripts/yolo_params.yaml   --out runs/report   --min-conf 0.25
```

**Or using classes.txt:**

```bash
python Hackathon2_scripts/generate_report.py   --pred predictions   --classes Hackathon2_scripts/ENV_SETUP/classes.txt   --out runs/report
```

**Report files (in `runs/report`):**
- `report_images.csv` — per-image detections  
- `report_classes.csv` — per-class count + avg confidence  
- `report_summary.txt` — human-readable summary  
- `report_bar.png` — bar chart of class distribution

---

## 🪪 Classes

| ID | Class Name         |
|----|--------------------|
| 0  | OxygenTank         |
| 1  | NitrogenTank       |
| 2  | FirstAidBox        |
| 3  | FireAlarm          |
| 4  | SafetySwitchPanel  |
| 5  | EmergencyPhone     |
| 6  | FireExtinguisher   |

---

## ⚡ Quick Start

```bash
# 1) Place dataset
#    GenIgnite/Hackathon2_test/

# 2) Install dependencies
pip install -U ultralytics matplotlib requests

# 3) Train
python Hackathon2_scripts/train.py --yaml Hackathon2_scripts/yolo_params.yaml

# 4) Predict
python Hackathon2_scripts/predict.py   --weights runs/detect/train_auto/weights/best.pt   --source Hackathon2_test/test/images

# 5) Report
python Hackathon2_scripts/generate_report.py   --pred predictions   --yaml Hackathon2_scripts/yolo_params.yaml   --out runs/report
```

---

## 📌 Notes

- Dataset is **not** included in the repo.  
- Folder name must be `Hackathon2_test` placed next to `Hackathon2_scripts`.  
- All paths are relative — no OS-specific hardcoded paths.  
- Compatible with low-VRAM GPUs (~4 GB) using auto-adjusted batch/img sizes.

---

## 🏁 Hackathon Context

Developed by **Team Tech Syndicate** for a hackathon focusing on:

- 🚀 Lightweight & portable deep-learning pipeline  
- ⚡ Real-time object detection (YOLOv8)  
- 📊 Automated reporting for easy evaluation

---

## 📜 License

- **Code:** Open for educational and research use.  
- **Dataset:** Not included — you must provide your own dataset under `Hackathon2_test/`.

---

## 🙌 Contributors

- Rahul Kala  
- Ravinder Singh

---
