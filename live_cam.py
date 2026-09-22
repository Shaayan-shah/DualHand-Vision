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

import argparse
from hand_detector import HandDetector
from visualizer import draw_hands_frame


def run_live(cam_index: int = 0, width: int = 1280, height: int = 720, mirror: bool = True, save_dir: str = None):
    print("Starting Live Both Hands Tracking & Finger Counter...")
    print("Controls: 'q'=Quit, 'p'=Save Snapshot")

    if save_dir is None:
        save_dir = os.path.join(MODULE_DIR, "data", "exports")
    os.makedirs(save_dir, exist_ok=True)

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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    fps = 0.0
    snapshot_idx = 0

    try:
        while cap.isOpened():
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Failed to read camera frame.")
                break

            if mirror:
                frame = cv2.flip(frame, 1)

            hands, summary = engine.process(frame, is_mirrored=mirror)

            elapsed = time.time() - t0
            fps = 1.0 / elapsed if elapsed > 0 else 0.0

            annotated = draw_hands_frame(frame, hands, summary, fps=fps)

            cv2.imshow("Both Hands & Finger Counter (Press Q to quit)", annotated)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('p'):
                snapshot_path = os.path.join(save_dir, f"snapshot_{int(time.time())}_{snapshot_idx:03d}.png")
                cv2.imwrite(snapshot_path, annotated)
                print(f"Saved snapshot to: {snapshot_path}")
                snapshot_idx += 1
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Live camera session ended.")


def parse_args():
    parser = argparse.ArgumentParser(description="Real-Time Dual-Hand Tracking & Finger Counter CLI")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Target video capture width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Target video capture height (default: 720)")
    parser.add_argument("--no-mirror", dest="mirror", action="store_false", help="Disable horizontal selfie mirroring")
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to save image snapshots")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_live(
        cam_index=args.camera,
        width=args.width,
        height=args.height,
        mirror=args.mirror,
        save_dir=args.save_dir
    )
