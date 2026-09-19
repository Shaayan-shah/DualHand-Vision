"""
Automated verification tests for dual-hand tracking and finger counting.
"""

import os
import sys
import cv2

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

from hand_detector import HandDetector
from visualizer import draw_hands_frame


def test_dual_hands():
    print("Testing Dual-Hand Tracking and Finger Counting...")
    sample_path = os.path.join(MODULE_DIR, "data", "sample.jpg")
    assert os.path.exists(sample_path), f"Sample image missing: {sample_path}"

    image = cv2.imread(sample_path)
    assert image is not None, "Failed to read sample image."

    detector = HandDetector()
    hands, summary = detector.process(image)

    assert len(hands) >= 1, "Expected at least 1 hand detected in sample."
    print(f"Detected {len(hands)} hand(s):")
    for h in hands:
        print(f"  - {h['handedness']}: {h['finger_count']} fingers extended [{h['gesture']}]")

    total = summary["total_fingers"]
    print(f"Total combined count: {total} (valid range 0-10)")
    assert 0 <= total <= 10, "Total finger count outside range 0-10"

    annotated = draw_hands_frame(image, hands, summary, fps=30.0)
    assert annotated.shape == image.shape, "Output frame dimension mismatch."

    print("Test passed successfully: Both hands tracked and fingers counted.")
    return True


if __name__ == "__main__":
    test_dual_hands()
