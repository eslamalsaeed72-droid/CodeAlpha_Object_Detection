# 👁️ Real-Time Object Detection & Tracking (YOLOv8 + Streamlit)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-green?style=for-the-badge&logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)

> **CodeAlpha AI Vision Internship Project**

## 📖 Overview

This repository hosts a robust **Computer Vision pipeline** designed for real-time object detection and tracking. Built with the power of **YOLOv8** (You Only Look Once) and **OpenCV**, wrapped in a user-friendly **Streamlit** dashboard.

The system is engineered to handle:
- 🔴 **Live Webcam Feed:** Low-latency inference for real-time monitoring.
- 📂 **Video File Processing:** Upload videos for frame-by-frame analysis and tracking.
- 🎯 **Object Tracking:** Implements centroid-based tracking to maintain object IDs across frames.

---

---

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| **State-of-the-Art Detection** | Utilizes `YOLOv8` (Medium/Nano) for high accuracy and speed. |
| **Smart Tracking** | Custom centroid tracking algorithm to trace object paths. |
| **Interactive UI** | Controlled via **Streamlit** allow users to toggle sources and adjust confidence thresholds dynamically. |
| **Development Notebook** | Includes `main_code.ipynb` for deep-dive experimentation and debugging. |

---

## 📂 Repository Structure

```text
├── app.py                  # 🚀 Main Streamlit Application
├── main_code.ipynb         # 📓 Jupyter Notebook (Development & Experiments)
├── requirements.txt        # 📦 Dependencies list
├── yolov8m.pt              # 🧠 Pre-trained YOLOv8 Model Weights
├── test/                   # 🧪 Test Assets
│   ├── video_test_1.mp4
│   └── video_test_2.mp4
├── Demo/                   # 🖼️ Demo Assets (Screenshots/GIFs)
│   └── screenshot.png
└── README.md               # 📄 Project Documentation

```

---

## 🛠️ Installation

Follow these steps to set up the environment locally:

**1. Clone the Repository**

```bash
git clone [https://github.com/eslamalsaeed72-droid/CodeAlpha_Object_Detection.git](https://github.com/eslamalsaeed72-droid/CodeAlpha_Object_Detection.git)
cd CodeAlpha_Object_Detection

```

**2. Create a Virtual Environment (Recommended)**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

```

**3. Install Dependencies**

```bash
pip install -r requirements.txt

```

> *Note: Ensure you have a CUDA-compatible GPU if you intend to run high-resolution real-time inference, though CPU is sufficient for testing.*

---

## 🚀 Usage

### 1. Running the Web App

Launch the Streamlit dashboard:

```bash
streamlit run app.py

```

* Open your browser at `http://localhost:8501`.
* **Sidebar Options:**
* Choose **"Live Webcam"** for real-time detection.
* Choose **"Video Upload"** to process `mp4`, `avi`, or `mov` files.
* Adjust the **Confidence Threshold** slider to filter weak detections.



### 2. Jupyter Notebook

To understand the tracking logic or retrain the model:

```bash
jupyter notebook main_code.ipynb

```

---

## 🏗️ Tech Stack

This project relies on the following key technologies:

* **Core:** Python 3.x
* **Deep Learning:** PyTorch, Ultralytics YOLOv8
* **Computer Vision:** OpenCV (`cv2`)
* **Web Framework:** Streamlit
* **Data Manipulation:** NumPy, Pandas

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<p align="center">
Developed by <strong>Eslam Alsaeed</strong> for CodeAlpha Internship
</p>

