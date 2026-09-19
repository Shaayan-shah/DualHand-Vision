"""
Live continuous webcam runner for Both Hands Tracking and Finger Counting.
Runs at 30+ FPS in an interactive desktop window.
Raise both hands, wave them, change fingers (0 to 10 total), and see real-time updates.
Press 'q' to exit, 'p' to save snapshot.
"""

import os
import sys
import time
import cv2

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
for p in [PARENT_DIR, MODULE_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from hand_detector import HandDetector
from visualizer import draw_hands_frame


def run_live(cam_index: int = 0):
    print("Starting Live Both Hands Tracking & Finger Counter...")
    print("Controls: 'q'=Quit, 'p'=Save Snapshot")

    engine = HandDetector()
    
    # Try DirectShow first on Windows for instant camera initialization
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_index)

    if not cap.isOpened():
        print(f"Error: Could not open camera at index {cam_index}. Attempting index 1...")
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: No functional webcam detected. Please check your camera permissions and connections.")
        return

    # Set preferred resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    fps = 0.0
    snapshot_idx = 0

    while cap.isOpened():
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Failed to read camera frame.")
            break

        # Flip horizontally for intuitive mirror view
        frame = cv2.flip(frame, 1)

        hands, summary = engine.process(frame, is_mirrored=True)

        elapsed = time.time() - t0
        fps = 1.0 / elapsed if elapsed > 0 else 0.0

        annotated = draw_hands_frame(frame, hands, summary, fps=fps)

        cv2.imshow("Both Hands & Finger Counter (Press Q to quit)", annotated)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('p'):
            export_dir = os.path.join(PARENT_DIR, "data", "exports")
            os.makedirs(export_dir, exist_ok=True)
            snapshot_path = os.path.join(export_dir, f"both_hands_snapshot_{snapshot_idx:03d}.png")
            cv2.imwrite(snapshot_path, annotated)
            print(f"Saved snapshot to: {snapshot_path}")
            snapshot_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Live camera session ended.")


if __name__ == "__main__":
    run_live()
