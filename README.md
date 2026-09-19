# Dual Hand Tracking and Finger Counter

A computer vision application that detects both hands simultaneously, tracks 21 three-dimensional skeletal landmarks per hand, counts extended fingers across a 0 to 10 range, and classifies static hand gestures in real time.

---

## Technical Overview

The system processes video input at 30+ frames per second using MediaPipe HandLandmarker and OpenCV. Unlike naive vertical threshold methods that fail when a hand is tilted or oriented horizontally, this implementation relies on rotation-invariant Euclidean distances calculated from anatomical anchor points.

### Key Capabilities

1. Simultaneous Dual Hand Detection: Identifies Left and Right hands independently and outputs individual (0 to 5) and combined (0 to 10) finger counts.
2. Rotation-Invariant Finger Counting: Uses distance comparisons between fingertip coordinates, intermediate joint positions, and the wrist base.
3. Mirror Mode Handedness Correction: Corrects selfie-camera mirroring so the user's physical right hand corresponds to the Right Hand label on screen.
4. Gesture Classification: Maps active finger combinations to common gestures including Closed Fist, Pointing, Peace / Victory, Three Fingers, Four Fingers, Open Palm, Thumbs Up, OK Sign, and Rock / Horns.
5. Dual Interface: Provides both an OpenCV direct desktop camera feed (for maximum frame rates) and an interactive Streamlit web dashboard.

---

## Project Structure

```
dual-hand-counter/
|-- app.py               # Streamlit web dashboard
|-- live_cam.py          # Real-time 30+ FPS camera feed
|-- hand_detector.py     # Hand detection and finger counting logic
|-- visualizer.py        # Landmark wireframe and HUD rendering
|-- test_counter.py      # Automated verification test script
|-- run.bat              # Windows batch launcher
|-- requirements.txt     # Python dependencies
|-- models/
|   `-- hand_landmarker.task   # MediaPipe task bundle
`-- data/
    `-- sample.jpg       # Photographic verification benchmark
```

---

## Finger Detection Logic

### Fingers 1 Through 4 (Index, Middle, Ring, Pinky)

For each finger with tip joint T, proximal interphalangeal joint P, metacarpophalangeal joint M, and wrist base W:

Distance(T, W) > Distance(P, W) and Distance(T, W) > Distance(M, W)

Because Euclidean distance from the wrist joint is invariant to 2D planar rotation, this condition holds regardless of whether the hand is vertical, tilted, or pointing sideways.

### Thumb Abduction

Because the thumb articulates sideways relative to the palm plane, radial extension is measured relative to the pinky MCP joint K:

Distance(ThumbTip, K) > 1.10 * Distance(ThumbIP, K)

When the thumb is tucked inward (as in a closed fist), its tip rests close to the palm, significantly reducing the distance to joint K.

---

## Installation

### Prerequisites

* Python 3.10, 3.11, or 3.12
* A functional USB or integrated webcam

### Setup

1. Open the project folder:
```bash
cd dual-hand-counter
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## How to Run

### 1. Interactive Menu (Windows)

Double-click run.bat or execute:
```cmd
run.bat
```

### 2. High-Speed Live Desktop Camera (30+ FPS)

```bash
python live_cam.py
```
* Press 'q' to exit the window.
* Press 'p' to capture and save a snapshot to disk.

### 3. Web Application (Streamlit)

```bash
streamlit run app.py
```

### 4. Run Automated Verification Tests

```bash
python test_counter.py
```

---

## License

MIT License.
