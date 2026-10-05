# MeterCrop

Automatic electricity meter detection and cropping using YOLOv8 + OpenCV.

![Python](https://img.shields.io/badge/python-3.12-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10-green)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

---

## Table of Contents

- [Problem](#problem)
- [Solution](#solution)
- [Pipeline](#pipeline)
- [Results](#results)
- [Dataset](#dataset)
- [Model](#model)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Sample Output](#sample-output)
- [Limitations](#limitations)
- [Next Steps](#next-steps)
- [Acknowledgements](#acknowledgements)

---

## Problem

Raw electricity meter photos collected from field surveys (HESCOM smart meters) contain a
significant amount of irrelevant content:

- GPS latitude/longitude text overlays
- Timestamps
- Surrounding walls, wooden boards, and mounting hardware
- Wires and terminals
- Plastic wrapping (in new installations)
- Neighboring switches and equipment

Running OCR directly on these full-resolution images is:

- **Slow** — OCR engines process many unnecessary pixels
- **Inaccurate** — the red GPS text and other content confuse the text detector
- **Storage-inefficient** — full-size images consume bandwidth and disk

## Solution

**MeterCrop** is a two-stage pipeline that automatically crops the meter region from a
raw photo:

1. **YOLOv8** detects the meter and returns a bounding box
2. **OpenCV** crops that region and saves a clean meter-only image

The result is a small, noise-free crop that is ideal for downstream OCR — the crop
typically covers **~10–15%** of the original pixel area, cutting processing cost by
**~85–90%**.

## Pipeline
┌─────────────┐ ┌──────────────────┐ ┌───────────────┐ ┌──────────────────┐
│ Raw photo │ -> │ YOLOv8 detection │ -> │ OpenCV crop │ -> │ Clean meter crop │
└─────────────┘ └──────────────────┘ └───────────────┘ └──────────────────┘
│
▼
(future) OCR reads
LCD display value

text

## Results

Evaluated on a held-out test set of **53 images** (images the model never saw during training):

| Metric              | Value     |
| ------------------- | --------- |
| **Test mAP50**      | **0.995** |
| **Test mAP50-95**   | **0.892** |
| Precision           | 0.998     |
| Recall              | 1.000     |
| Inference speed     | ~8 ms/image (Tesla T4) |
| Model size          | ~6 MB     |

**Interpretation:**

- **mAP50 = 0.995** — near-perfect detection at IoU ≥ 0.5
- **mAP50-95 = 0.892** — tight bounding boxes even at strict IoU thresholds
- **Recall = 1.000** — every real meter in the test set was detected
- **Precision = 0.998** — virtually no false positives

## Dataset

| Property | Value |
| -------- | ----- |
| Total images | 356 |
| Source | Field-collected HESCOM smart meter photos |
| Annotation tool | [Roboflow](https://roboflow.com) |
| Classes | 1 (`meter`) |
| Train / Valid / Test split | 70% / 15% / 15% |
| Training images (after augmentation) | 747 |
| Validation images | 54 |
| Test images | 53 |

**Augmentation** (applied to training set only):

- Horizontal flip
- Rotation ±15°
- Brightness ±25%
- Blur ≤ 2 px

**Annotation rules** used:

- Full meter visible → tight box around the meter body
- Partial meter (>30% visible) → box the visible portion only
- Meter barely visible (<10%) → left unlabeled (acts as a negative sample)
- No meter → left unlabeled (negative sample)

## Model

| Property | Value |
| -------- | ----- |
| Architecture | YOLOv8-nano (`yolov8n.pt`) |
| Library | [Ultralytics](https://github.com/ultralytics/ultralytics) 8.4 |
| Epochs | 50 |
| Image size | 640 × 640 |
| Batch size | 16 |
| Optimizer | AdamW (auto-selected) |
| Early stopping patience | 15 epochs |
| Hardware | Kaggle GPU — Tesla T4 |
| Training time | ~5 minutes |

## Project Structure
meter-crop/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── notebook/
│ └── meter_detection.ipynb # full training + evaluation notebook
│
├── model/
│ └── best.pt # trained YOLOv8 model (~6 MB)
│
├── scripts/
│ ├── crop_images.py # standalone cropping script
│ └── train.py # standalone training script (optional)
│
├── sample_crops/
│ ├── crop_01.png
│ ├── crop_02.png
│ └── crop_03.png
│
└── docs/
├── results.png # training curves
├── confusion_matrix.png
└── sample_prediction.png

text

## Setup

### Requirements

- Python 3.10+
- pip

### Install

```bash
git clone https://github.com/<your-username>/meter-crop.git
cd meter-crop
pip install -r requirements.txt
Requirements File
text
ultralytics>=8.4.0
opencv-python>=4.10.0
matplotlib>=3.10.0
pillow>=11.3.0
Usage
1. Crop a single image
python
from ultralytics import YOLO
import cv2

model = YOLO("model/best.pt")

img = cv2.imread("meter_photo.jpg")
results = model(img, conf=0.5, verbose=False)[0]

if len(results.boxes) == 0:
    print("No meter detected")
else:
    x1, y1, x2, y2 = map(int, results.boxes[0].xyxy[0])

    # add 5% margin to avoid clipping meter edges
    mx = int((x2 - x1) * 0.05)
    my = int((y2 - y1) * 0.05)
    x1, y1 = max(0, x1 - mx), max(0, y1 - my)
    x2, y2 = min(img.shape[1], x2 + mx), min(img.shape[0], y2 + my)

    crop = img[y1:y2, x1:x2]
    cv2.imwrite("meter_crop.jpg", crop)
    print(f"Saved crop ({x2-x1}×{y2-y1}px)")
2. Batch crop a folder
python
import cv2, json, os
from pathlib import Path
from ultralytics import YOLO

model = YOLO("model/best.pt")

INPUT_DIR  = Path("images")
OUTPUT_DIR = Path("crops")
OUTPUT_DIR.mkdir(exist_ok=True)

log = []
for img_path in sorted(INPUT_DIR.glob("*")):
    if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        continue

    h, w = img.shape[:2]
    results = model(img, conf=0.5, verbose=False)[0]

    if len(results.boxes) == 0:
        print(f"⚠️  no meter in {img_path.name}")
        continue

    x1, y1, x2, y2 = map(int, results.boxes[0].xyxy[0])
    mx, my = int((x2 - x1) * 0.05), int((y2 - y1) * 0.05)
    x1, y1 = max(0, x1 - mx), max(0, y1 - my)
    x2, y2 = min(w, x2 + mx), min(h, y2 + my)

    cv2.imwrite(str(OUTPUT_DIR / img_path.name), img[y1:y2, x1:x2])
    log.append({
        "file": img_path.name,
        "box":  [x1, y1, x2, y2],
        "conf": float(results.boxes[0].conf[0])
    })

with open(OUTPUT_DIR / "boxes.json", "w") as f:
    json.dump(log, f, indent=2)

print(f"✅ Cropped {len(log)} images")
3. Tune the confidence threshold
The conf parameter controls how strict the detector is:

Value	Behavior
0.2	Very lenient — catches more, more false positives
0.5	Balanced — recommended default
0.8	Very strict — misses harder cases
Sample Output
Here is an example: a raw photo with GPS text, wall, wires, and plastic wrap, followed by
the clean crop produced by the model.

Original (noisy)	Cropped (clean)
https://sample_crops/original.png	https://sample_crops/crop_01.png
Limitations
Meters too small (<5% of image area) may be missed — the model was trained on
reasonably-sized meters in frame.

Heavy plastic wrap / strong reflections may reduce detection confidence.

Multi-meter panels — the current script only crops the highest-confidence box.
Extend the script to loop over results.boxes for multiple meters.

Images with no meter — the model correctly outputs zero boxes; handle in code.

Grayscale training data — if the source images were grayscale, crops remain
grayscale. This does not affect OCR accuracy for Tesseract/PaddleOCR.

Next Steps
□ OCR integration — feed crops into Tesseract or PaddleOCR to read the LCD
□ Multi-meter support — crop all detected meters per image
□ API deployment — expose as a REST endpoint with FastAPI
□ Expand dataset — add more images to improve robustness on rare conditions
□ Fine-tune on hard cases — images with heavy blur or obstruction
□ Model variants — compare YOLOv8s / YOLOv8m for higher accuracy
Acknowledgements
Ultralytics YOLOv8 — detection framework

Roboflow — annotation + dataset versioning

OpenCV — image processing

Kaggle — free GPU training environment

License
MIT — see LICENSE for details.

Contact
Author: <Your Name>

Project Repository: github.com/<your-username>/meter-crop

text

---

## How to Use This README

1. **Create the file** — save as `README.md` at the **root** of your GitHub repo
2. **Replace placeholders**:
   - `<your-username>` → your GitHub username (4 places)
   - `<Your Name>` → your actual name in the Contact section
3. **Copy the sample images** into `sample_crops/`:
   - Name them `crop_01.png`, `crop_02.png`, etc.
   - Optionally add `original.png` (an uncropped original) for the comparison table
4. **Copy training plots** into `docs/`:
   - `results.png` from `runs/detect/meter_yolov8n/`
   - `confusion_matrix.png` from same folder
   - Optional: `sample_prediction.png` (a YOLO-labelled test image)

---

## Optional Additions

### If you want a "demo" GIF

Replace the sample table with:
```markdown
![Demo](docs/demo.gif)
Then record a short screen capture of running the cropping script.

If you want to add a Colab badge
At the top:

markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/meter-crop/blob/main/notebook/meter_detection.ipynb)
If you're deploying it
Add a Deployment section:

markdown
## Deployment

### Docker

```bash
docker build -t meter-crop .
docker run -p 8000:8000 meter-crop
text

---

## Checklist Before Committing the README

- [ ] All `<placeholders>` replaced
- [ ] Sample crops in `sample_crops/`
- [ ] Training plots in `docs/`
- [ ] Repository URL in Contact section is correct
- [ ] License file exists (add `LICENSE` with MIT text)
- [ ] Requirements file exists
- [ ] All image paths in the README match actual file locations

---

## `LICENSE` File (MIT — Paste Alongside README)
MIT License

Copyright (c) 2026 <Your Name>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

text

---

## Final Steps

1. Save the README as `README.md`
2. Save the LICENSE as `LICENSE`
3. Organize your repo folder as shown in "Project Structure"
4. Run `git add . && git commit -m "Add README and license" && git push`

**Tell me when it's pushed** — then I'll help with OCR integration or deployment if you want.

