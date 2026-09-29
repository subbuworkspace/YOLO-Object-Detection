# 🚀 YOLO Object Detection using Python

A beginner-friendly **Computer Vision project using YOLO (You Only Look Once)** for detecting objects in images, videos, and real-time webcam streams.

This project is designed to understand how modern object detection works using a pretrained YOLO model and Python.

---

## 📌 Project Overview

**YOLO (You Only Look Once)** is a popular real-time object detection algorithm.

Unlike traditional image classification, YOLO can identify:

* **What object is present**
* **Where the object is located**
* **How confident the model is**

For example:

```text
Input Image
     ↓
YOLO Model
     ↓
Object Detection
     ↓
┌─────────────────────┐
│  Person   95%       │
│                     │
│  Car      91%       │
│                     │
│  Dog      87%       │
└─────────────────────┘
```

---

# 🎯 Project Objectives

The main objectives of this project are:

* Understand YOLO object detection.
* Run a pretrained YOLO model.
* Detect multiple objects in an image.
* Perform real-time webcam detection.
* Understand bounding boxes.
* Understand confidence scores.
* Work with OpenCV.
* Save and visualize detection results.
* Build a foundation for custom object detection.

---

# 🧠 What is YOLO?

**YOLO = You Only Look Once**

YOLO is a deep-learning-based object detection approach designed for real-time computer vision.

The basic idea is:

```text
Image
  ↓
YOLO Neural Network
  ↓
Object Detection
  ↓
Bounding Boxes
  ↓
Class + Confidence
```

For example:

```text
Person → 96%
Car    → 91%
Dog    → 88%
```

The model processes the image and predicts objects and their locations in a single inference pipeline.

---

# 🔍 Object Detection vs Image Classification

## Image Classification

Classification answers:

> "What is in this image?"

Example:

```text
Image
  ↓
Cat
```

---

## Object Detection

Object detection answers:

> "What objects are present and where are they?"

Example:

```text
Image
  ↓
┌──────────────┐
│     CAT      │
│     96%      │
└──────────────┘

┌──────────────┐
│     DOG      │
│     91%      │
└──────────────┘
```

---

# ⭐ Why YOLO?

YOLO is widely used because it provides a practical balance between:

* Speed
* Accuracy
* Real-time inference
* Multiple object detection
* Easy deployment

It can be used for:

* CCTV analytics
* Traffic monitoring
* Vehicle detection
* People detection
* Retail analytics
* Manufacturing inspection
* Robotics
* Sports analytics
* Safety monitoring

---

# 🛠️ Technologies Used

| Technology  | Purpose                |
| ----------- | ---------------------- |
| Python      | Programming            |
| YOLO        | Object Detection       |
| Ultralytics | YOLO implementation    |
| OpenCV      | Image/video processing |
| NumPy       | Numerical operations   |

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/YOLO-Object-Detection.git
```

Navigate to the project:

```bash
cd YOLO-Object-Detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 📄 requirements.txt

```text
ultralytics
opencv-python
```

---

# 📁 Project Structure

```text
YOLO-Object-Detection/
│
├── README.md
├── requirements.txt
│
├── src/
│   ├── image_detection.py
│   ├── video_detection.py
│   └── webcam_detection.py
│
├── images/
│   ├── input/
│   └── output/
│
├── results/
│
└── models/
```

---

# 🖼️ 1. Image Object Detection

Load a pretrained YOLO model:

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
```

Provide an image:

```python
image_path = "images/input/test.jpg"
```

Run prediction:

```python
results = model(image_path)
```

Display the result:

```python
for result in results:
    result.show()
```

Save the result:

```python
for result in results:
    result.save("images/output/detected.jpg")
```

---

# 🎥 2. Video Object Detection

YOLO can also process video files.

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

results = model(
    "video.mp4",
    save=True
)
```

The model detects objects frame by frame.

```text
Video
  ↓
Frame 1 → YOLO
Frame 2 → YOLO
Frame 3 → YOLO
Frame 4 → YOLO
       ↓
Detection Results
```

---

# 📷 3. Real-Time Webcam Detection

Open the webcam using OpenCV:

```python
import cv2
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)

    annotated_frame = results[0].plot()

    cv2.imshow(
        "YOLO Object Detection",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

Press:

```text
Q
```

to stop the webcam.

---

# 📦 Bounding Boxes

A bounding box represents the location of an object.

Conceptually:

```text
(x1, y1) ───────────────
   │                    │
   │       PERSON       │
   │                    │
   ─────────────────────
                  (x2,y2)
```

YOLO predicts the coordinates of the detected object.

---

# 🎯 Confidence Score

The confidence score represents how confident the model is about its prediction.

Example:

```text
Person → 0.96
Car    → 0.91
Dog    → 0.87
```

Usually displayed as:

```text
Person 96%
Car    91%
Dog    87%
```

A higher value indicates greater model confidence, although confidence should not be interpreted as a guaranteed probability of correctness.

---

# 🧩 YOLO Detection Pipeline

```text
                 Input
                   │
                   ▼
              Image / Video
                   │
                   ▼
              YOLO Model
                   │
                   ▼
             Object Detection
                   │
          ┌────────┴────────┐
          ▼                 ▼
      Bounding Box       Class
          │                 │
          └────────┬────────┘
                   ▼
             Confidence
                   │
                   ▼
             Final Result
```

---

# 🧠 Important YOLO Concepts

## 1. Bounding Box

Defines where the object is located.

## 2. Class

Identifies what the object is.

Example:

```text
Person
Car
Dog
Cat
Bus
Truck
```

## 3. Confidence

Indicates the model's confidence in the prediction.

## 4. IoU

**Intersection over Union (IoU)** measures the overlap between two bounding boxes.

```text
IoU = Area of Intersection
      --------------------
      Area of Union
```

It is important in object detection evaluation and Non-Maximum Suppression.

---

# 🚫 Non-Maximum Suppression

Sometimes the model may produce multiple overlapping bounding boxes for the same object.

Example:

```text
       ┌─────────────┐
       │    PERSON   │
       └─────────────┘
      ┌───────────────┐
      │    PERSON    │
      └───────────────┘
```

Non-Maximum Suppression helps retain the more appropriate detection and remove redundant overlapping boxes.

---

# 📊 Object Detection Metrics

Common evaluation metrics include:

### Precision

How many predicted positive detections were actually correct.

### Recall

How many actual objects were successfully detected.

### IoU

Measures overlap between predicted and ground-truth bounding boxes.

### mAP

**Mean Average Precision**

A widely used object detection evaluation metric.

---

# 🏗️ Pretrained Model

This project uses a pretrained YOLO model.

```python
model = YOLO("yolo26n.pt")
```

The smaller `n` model is useful for learning and experimentation because it is designed to be lightweight.

For production or higher-accuracy requirements, different model sizes can be evaluated based on the required speed and accuracy trade-off.

---

# 🔥 Project Features

* ✅ Image detection
* ✅ Video detection
* ✅ Real-time webcam detection
* ✅ Multiple object detection
* ✅ Bounding boxes
* ✅ Class labels
* ✅ Confidence scores
* ✅ OpenCV integration
* ✅ Detection result visualization
* ✅ Foundation for custom training

---

# 🚀 Future Enhancements

This project can be extended significantly.

## Phase 1

```text
YOLO
 ↓
Image Detection
 ↓
Webcam Detection
```

## Phase 2

```text
YOLO
 ↓
Video Detection
 ↓
Object Tracking
```

## Phase 3

```text
Custom Dataset
       ↓
Data Annotation
       ↓
YOLO Training
       ↓
Custom Object Detector
```

## Phase 4

Build a frontend:

```text
YOLO Model
     ↓
Python
     ↓
Streamlit
     ↓
Web Application
```

The Streamlit application could allow users to upload an image or video and view the detection results.

---

# 💡 Possible Real-World Projects

After completing this basic project, the same approach can be used for:

### 🚗 Number Plate Detection

```text
Vehicle
   ↓
YOLO
   ↓
Number Plate
   ↓
OCR
   ↓
Registration Number
```

### 🚦 Traffic Detection

```text
Camera
 ↓
YOLO
 ↓
Cars / Bikes / Buses
 ↓
Traffic Analytics
```

### 👷 Safety Detection

```text
Camera
 ↓
YOLO
 ↓
Person
 ↓
Helmet / Safety Equipment
 ↓
Safety Monitoring
```

### 🏭 Industrial Inspection

```text
Camera
 ↓
YOLO
 ↓
Defect Detection
 ↓
Quality Control
```

---

# 🎤 Interview Questions

## Beginner

### 1. What is YOLO?

YOLO stands for **You Only Look Once** and is used for object detection.

### 2. What is object detection?

Object detection identifies objects and their locations within an image or video.

### 3. What is the difference between classification and detection?

Classification identifies the image or class, while detection identifies multiple objects and their locations.

### 4. What is a bounding box?

A bounding box represents the location of a detected object.

### 5. What is confidence score?

It indicates how confident the model is about a particular detection.

---

## Intermediate

### 6. What is IoU?

IoU measures the overlap between a predicted bounding box and a ground-truth bounding box.

### 7. What is NMS?

Non-Maximum Suppression removes redundant overlapping detections.

### 8. What is mAP?

mAP stands for Mean Average Precision and is commonly used to evaluate object detection performance.

### 9. Why is YOLO considered suitable for real-time detection?

YOLO performs object detection efficiently enough for many real-time computer vision applications.

### 10. What is transfer learning?

Transfer learning uses a model pretrained on a large dataset as the starting point for another task.

---

# 📚 Learning Flow

```text
Python
  ↓
OpenCV
  ↓
Computer Vision
  ↓
CNN Basics
  ↓
Object Detection
  ↓
YOLO
  ↓
YOLO + OpenCV
  ↓
Custom Dataset
  ↓
Custom YOLO Training
  ↓
Streamlit Application
```

---

# 🎯 Learning Outcome

After completing this project, you should understand:

* What YOLO is
* How object detection works
* Bounding boxes
* Classes
* Confidence scores
* IoU
* NMS
* mAP
* Image detection
* Video detection
* Webcam detection
* OpenCV + YOLO integration
* The basic workflow for custom object detection

---

# 👨‍💻 Author

**Subrata Mondal**

Python | Data Analytics | Data Science | Machine Learning | Computer Vision | AI

---

## ⭐ Future Goal

> **Learn the fundamentals → Build the project → Train on a custom dataset → Deploy with Streamlit**

⭐ If you find this project useful, consider giving the repository a star!
