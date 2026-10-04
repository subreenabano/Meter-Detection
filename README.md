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
