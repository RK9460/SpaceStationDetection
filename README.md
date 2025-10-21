# 🚀 Space Station Safety Detection — YOLOv8

A lightweight and portable **YOLOv8-based object detection pipeline** to detect critical safety equipment aboard a space station environment or industrial setup.

---

## 👥 Team Information

- **Team Name:** Tech Syndicate  
- **Team Members:**  
  - Rahul Kala  
  - Ravinder Singh

---

## 📁 Project Structure

GenIgnite/
├─ README.md
├─ Hackathon2_scripts/
│ ├─ train.py
│ ├─ predict.py
│ ├─ generate_report.py
│ ├─ visualize.py
│ ├─ yolo_params.yaml
│ └─ ENV_SETUP/
│ └─ classes.txt
│
└─ Hackathon2_test/ # 📂 Place your dataset here (not in repo)
├─ train/
│ ├─ images/
│ └─ labels/
├─ val/
│ ├─ images/
│ └─ labels/
└─ test/
├─ images/
└─ labels/


⚠️ **Note:**  
The dataset (`Hackathon2_test`) is **not included** in the repository.  
Each user or evaluator must manually place their dataset in this folder.

---

## 🧰 Requirements

- Python 3.10+
- GPU optional (CUDA recommended)
- Install dependencies:

```bash
pip install -U ultralytics matplotlib requests


(For GPU acceleration, install the correct CUDA-enabled PyTorch from https://pytorch.org
).

🧠 Dataset Configuration

Hackathon2_scripts/yolo_params.yaml

path: ../Hackathon2_test
train: train/images
val:   val/images
test:  test/images

nc: 7
names: ['OxygenTank','NitrogenTank','FirstAidBox','FireAlarm','SafetySwitchPanel','EmergencyPhone','FireExtinguisher']

Folder Format:
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


Each .txt file (YOLO label) contains:

<class_id> <x_center> <y_center> <width> <height>


All coordinates are normalized between 0 and 1.

🧪 Training
python Hackathon2_scripts/train.py \
  --yaml Hackathon2_scripts/yolo_params.yaml \
  --weights weights/yolov8n.pt \
  --results runs/detect


✅ Auto-downloads weights if missing
✅ Auto-adjusts image & batch size for low-VRAM GPUs
✅ Saves results in runs/detect/train_auto/

🔍 Prediction (Inference)
📁 Predict on entire test set
python Hackathon2_scripts/predict.py \
  --weights runs/detect/train_auto/weights/best.pt \
  --source Hackathon2_test/test/images \
  --out-images predictions/images \
  --out-labels predictions/labels \
  --conf 0.25 --imgsz 640 --iou 0.45 --augment

🖼️ Predict on single image
python Hackathon2_scripts/predict.py \
  --weights runs/detect/train_auto/weights/best.pt \
  --source Hackathon2_test/test/images/example.png


📤 Output:

Annotated images → predictions/images

YOLO-format label files (with confidence) → predictions/labels

📊 Report Generation

Using YAML:

python Hackathon2_scripts/generate_report.py \
  --pred predictions \
  --yaml Hackathon2_scripts/yolo_params.yaml \
  --out runs/report \
  --min-conf 0.25


Using classes.txt:

python Hackathon2_scripts/generate_report.py \
  --pred predictions \
  --classes Hackathon2_scripts/ENV_SETUP/classes.txt \
  --out runs/report


📈 Output (runs/report/):

report_images.csv     # per image
report_classes.csv    # per class count + avg confidence
report_summary.txt    # human-readable summary
report_bar.png        # bar chart

🪪 Classes
ID	Class Name
0	OxygenTank
1	NitrogenTank
2	FirstAidBox
3	FireAlarm
4	SafetySwitchPanel
5	EmergencyPhone
6	FireExtinguisher
⚡ Quick Start
# 1. Place dataset at:
#    GenIgnite/Hackathon2_test/

# 2. Install dependencies
pip install -U ultralytics matplotlib requests

# 3. Train
python Hackathon2_scripts/train.py --yaml Hackathon2_scripts/yolo_params.yaml

# 4. Predict
python Hackathon2_scripts/predict.py \
  --weights runs/detect/train_auto/weights/best.pt \
  --source Hackathon2_test/test/images

# 5. Report
python Hackathon2_scripts/generate_report.py \
  --pred predictions \
  --yaml Hackathon2_scripts/yolo_params.yaml \
  --out runs/report

📌 Notes

No dataset is committed to the repo.

Dataset folder must be named Hackathon2_test and placed next to Hackathon2_scripts.

All paths are relative — no hardcoded Windows paths.

Compatible with low-VRAM GPUs (e.g., 4 GB).

🏁 Hackathon Context

This project was developed by Team Tech Syndicate for a hackathon challenge focusing on:

🚀 Lightweight & portable deep learning pipeline

⚡ Real-time object detection (YOLOv8)

📊 Automated reporting for evaluation

📜 License

Code: Open for educational and research use.

Dataset: Not included. Users must provide their own dataset under Hackathon2_test/.

🙌 Contributors

Rahul Kala

Ravinder Singh
