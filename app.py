import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# -------------------------------
# Configuration and model loading
# -------------------------------

COMPANY_NAME = "CodeAlpha AI Vision"
DEFAULT_MODEL_PATH = "yolov8m.pt"
DEFAULT_CONFIDENCE = 0.5


@st.cache_resource
def load_model(model_path: str = DEFAULT_MODEL_PATH) -> YOLO:
    """
    Load a pretrained YOLO model once and cache it for reuse.

    Parameters
    ----------
    model_path : str
        Path to the YOLO weights file.

    Returns
    -------
    YOLO
        Loaded YOLO model instance.
    """
    model = YOLO(model_path)
    return model


model = load_model()


# -------------------------------
# Utility functions
# -------------------------------

def compute_centroid(box):
    """
    Compute the centroid of a bounding box.

    Parameters
    ----------
    box : list or tuple
        Bounding box coordinates [x1, y1, x2, y2].

    Returns
    -------
    np.ndarray
        Centroid coordinates (cx, cy).
    """
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return np.array([cx, cy], dtype=np.float32)


def associate_detections_to_tracks(detections, prev_positions, distance_threshold=50.0):
    """
    Associate detections with existing tracks using Euclidean distance.

    Parameters
    ----------
    detections : list
        List of (box, confidence, class_name).
    prev_positions : dict
        Mapping track_id -> last centroid position.
    distance_threshold : float
        Maximum allowed distance to keep the same track.

    Returns
    -------
    dict
        Mapping track_id -> (box, confidence, class_name, centroid).
    """
    assignments = {}
    used_tracks = set()
    next_track_id = 0

    if prev_positions:
        next_track_id = max(prev_positions.keys()) + 1

    for box, confidence, class_name in detections:
        centroid = compute_centroid(box)
        best_track_id = None
        best_distance = distance_threshold

        for track_id, prev_centroid in prev_positions.items():
            if track_id in used_tracks:
                continue

            distance = np.linalg.norm(centroid - prev_centroid)
            if distance < best_distance:
                best_distance = distance
                best_track_id = track_id

        if best_track_id is not None:
            assignments[best_track_id] = (box, confidence, class_name, centroid)
            used_tracks.add(best_track_id)
        else:
            assignments[next_track_id] = (box, confidence, class_name, centroid)
            next_track_id += 1

    return assignments


def run_inference_on_frame(frame, confidence_threshold):
    """
    Run YOLO inference on a single frame and return detections.

    Parameters
    ----------
    frame : np.ndarray
        Input frame in BGR format.
    confidence_threshold : float
        Minimum required confidence.

    Returns
    -------
    list
        List of (box, confidence, class_name).
    """
    results = model(frame, verbose=False)
    detections = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            class_name = model.names[class_id]

            if confidence < confidence_threshold:
                continue

            detections.append(
                (
                    [int(x1), int(y1), int(x2), int(y2)],
                    confidence,
                    class_name,
                )
            )

    return detections


def annotate_frame_with_tracks(frame, assignments, track_positions):
    """
    Draw bounding boxes, labels, and track IDs on a frame.

    Parameters
    ----------
    frame : np.ndarray
        Input frame in BGR format (modified in place).
    assignments : dict
        Mapping track_id -> (box, confidence, class_name, centroid).
    track_positions : dict
        Mapping track_id -> centroid (updated in place).

    Returns
    -------
    np.ndarray
        Annotated frame in BGR format.
    """
    track_positions.clear()

    for track_id, (bbox, confidence, class_name, centroid) in assignments.items():
        track_positions[track_id] = centroid

        x1, y1, x2, y2 = bbox
        color = (0, 180, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID {track_id} | {class_name} {confidence:.2f}"

        cv2.putText(
            frame,
            label,
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            lineType=cv2.LINE_AA,
        )

        cv2.circle(frame, tuple(centroid.astype(int)), 4, (0, 255, 0), -1)

    return frame


def process_video_file(input_path: Path, output_path: Path, confidence_threshold: float):
    """
    Run detection and tracking on a video file and save the annotated output.

    Parameters
    ----------
    input_path : Path
        Path to the input video.
    output_path : Path
        Path to the output video.
    confidence_threshold : float
        Minimum required confidence.
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {input_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

    track_positions = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = run_inference_on_frame(frame, confidence_threshold)
        assignments = associate_detections_to_tracks(detections, track_positions)
        frame = annotate_frame_with_tracks(frame, assignments, track_positions)

        writer.write(frame)

    cap.release()
    writer.release()


# -------------------------------
# Streamlit layout and styling
# -------------------------------

st.set_page_config(
    page_title=f"{COMPANY_NAME} – Object Detection",
    page_icon="🎯",
    layout="wide",
)

# Custom header
st.markdown(
    f"""
    <div style="background: linear-gradient(90deg, #2b5876, #4e4376); padding: 1.2rem 1.5rem; border-radius: 0.5rem; margin-bottom: 1.5rem;">
        <h1 style="color: #ffffff; margin: 0; font-size: 1.8rem;">{COMPANY_NAME}</h1>
        <p style="color: #e0e0e0; margin: 0.2rem 0 0;">
            Real-Time Object Detection and Tracking – YOLO + OpenCV
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.title("Control Panel")
st.sidebar.markdown("Configure detection settings and input source.")

confidence = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.1,
    max_value=0.9,
    value=DEFAULT_CONFIDENCE,
    step=0.05,
)

mode = st.sidebar.radio(
    "Select mode",
    options=["Live Webcam", "Video Upload"],
)

st.sidebar.markdown("---")
st.sidebar.write("Developed for AI Vision demos.")

# Main layout
col_left, col_right = st.columns([3, 2])

with col_right:
    st.markdown("### Status")
    st.info(
        "Choose **Live Webcam** for real-time detection from your camera, "
        "or **Video Upload** to process a recorded clip."
    )

# -------------------------------
# Live webcam mode
# -------------------------------
if mode == "Live Webcam":
    st.markdown("## Live Webcam")
    st.write("Press **Start** to begin streaming from your webcam. Press **Stop** or close the tab to end.")

    start_button = st.button("Start webcam")

    if start_button:
        frame_placeholder = st.empty()
        info_placeholder = st.empty()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Failed to access webcam. Check your camera or permissions.")
        else:
            track_positions = {}
            info_placeholder.info("Streaming from webcam. Close the browser tab or press `q` in the OpenCV window to stop.")

            while True:
                ret, frame = cap.read()
                if not ret:
                    st.warning("No frame received from webcam.")
                    break

                detections = run_inference_on_frame(frame, confidence)
                assignments = associate_detections_to_tracks(detections, track_positions)
                frame = annotate_frame_with_tracks(frame, assignments, track_positions)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

                # Optional escape route if user closes server-side loop via keyboard
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            cap.release()
            cv2.destroyAllWindows()
            info_placeholder.info("Webcam stream stopped.")

# -------------------------------
# Video upload mode
# -------------------------------
elif mode == "Video Upload":
    st.markdown("## Video Upload")
    st.write("Upload a video file to run detection and tracking, then preview the result.")

    uploaded_video = st.file_uploader(
        "Upload a video file", type=["mp4", "mov", "avi", "mkv"]
    )

    if uploaded_video is not None:
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_input.write(uploaded_video.read())
        temp_input.close()

        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_output.close()

        st.video(temp_input.name)

        if st.button("Run detection and tracking"):
            with st.spinner("Processing video. Please wait..."):
                process_video_file(
                    input_path=Path(temp_input.name),
                    output_path=Path(temp_output.name),
                    confidence_threshold=confidence,
                )

            st.success("Processing complete.")
            st.markdown("### Processed video")
            st.video(temp_output.name)
