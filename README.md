# Real-Time Object Detection and Tracking (YOLOv8 + Streamlit)

This repository contains a complete real-time object detection and tracking pipeline
built with **YOLOv8**, **OpenCV**, and **Streamlit**. The system can run on:

- Live webcam feed.
- Uploaded video files.
- Jupyter Notebook for experimentation and development.

The project is designed as a clean, extensible reference implementation for
computer vision applications such as surveillance, traffic monitoring, and general
object-tracking demos.

---

## 1. Project Features

- Real-time object detection using a pretrained **YOLOv8** model.
- Basic multi-object tracking using a centroid-based tracking strategy.
- Web interface built with **Streamlit**:
  - Live webcam mode.
  - Video upload mode with processed output.
- Jupyter Notebook (`main_code.ipynb`) for step-by-step exploration.
- Sample test videos and demo assets included.

---

## 2. Repository Structure

```text
.
├── app.py                 # Streamlit web application
├── main_code.ipynb        # Jupyter Notebook (detection + tracking pipeline)
├── requirements.txt       # Python dependencies
├── yolov8m.pt             # Pretrained YOLOv8 model weights
├── README.md              # Project documentation (this file)
├── LICENSE                # Project license (MIT)
│
├── test/                  # Test assets
│   ├── video_test_1.mp4   # Sample test video 1
│   └── video_test_2.mp4   # Sample test video 2
│
└── Demo/                  # Demo artifacts
    ├── demo_video.mp4     # Demo video showing the system in action
    └── screenshot.png     # Screenshot of the application UI
```

> Note: The exact filenames under `test/` and `Demo/` can be adjusted as needed,  
> but the folder structure is kept consistent for clarity.

---

## 3. Installation

1. Clone or download this repository.

2. Create and activate a virtual environment (recommended):

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

Make sure your environment has a compatible version of **PyTorch** with GPU support
if you plan to run real-time inference on high-resolution video.

---

## 4. Running the Streamlit App

The `app.py` file exposes a simple web interface with two main modes:
**Live Webcam** and **Video Upload**.

```bash
streamlit run app.py
```

Once the app is running:

- Open the URL shown in the terminal (usually `http://localhost:8501`).
- Use the sidebar to:
  - Select the mode: `Live Webcam` or `Video Upload`.
  - Adjust the confidence threshold for detections.
- In **Live Webcam** mode:
  - Start the webcam stream and observe real-time detection and tracking.
- In **Video Upload** mode:
  - Upload a video file.
  - Run detection and tracking.
  - Preview the processed result directly in the browser.

The header shows the project identity:  
`CodeAlpha AI Vision – Real-Time Object Detection and Tracking`.

---

## 5. Jupyter Notebook (`main_code.ipynb`)

The notebook contains a step-by-step implementation of:

1. Environment checks and imports.
2. Model loading and configuration (YOLOv8).
3. Utility functions for centroid tracking.
4. Real-time webcam detection and tracking loop.
5. Video file processing and result export.

It is useful for:
- Understanding the underlying pipeline.
- Experimenting with different model variants (e.g., `yolov8n.pt` vs `yolov8m.pt`).
- Customizing the tracking logic or visualization.

---

## 6. Testing

You can quickly validate the pipeline using the videos in the `test/` folder:

1. Start the Streamlit app.
2. Switch to **Video Upload** mode.
3. Upload one of the test videos, for example:
   - `test/video_test_1.mp4`
   - `test/video_test_2.mp4`
4. Run detection and tracking and inspect the resulting video.

The `Demo/` folder includes a pre-generated demo video and screenshots to show
expected behavior and UI layout.

---

## 7. Technologies and Tags

Core technologies and concepts used in this project:

- **YOLOv8**, **Ultralytics**, **Object Detection**
- **Multi-Object Tracking**, **Centroid Tracking**
- **OpenCV**, **Computer Vision**
- **Streamlit**, **Web App**, **Real-Time Inference**
- **Python**, **PyTorch**, **Deep Learning**
- **Webcam Processing**, **Video Analytics**, **AI Demo**

These keywords describe the main focus areas and can be used as tags for
GitHub, portfolio entries, or documentation.

---

## 8. License

This project is released under the **MIT License**.  
You are free to use, modify, and distribute the code, subject to the terms
described in the `LICENSE` file.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/connectors/google_drive/1lkjnZtlO8D2XAYZ1sIml6-dqB6ItYxk0sKDpcyw4l-w/e3c0a92d-5860-4704-9b1a-e098d7c7b469/Artificial-Intelligence-Tasks-Instructions-CodeAlpha.md)
[2](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/157424363/7ce71c98-da18-431d-bb50-54a1810aa2df/image.jpg)
[3](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/157424363/df8042ce-7c16-4c0e-a27c-c295c42023a8/image.jpg)