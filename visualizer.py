"""
Visual overlays for Both Hands Tracking and Finger Counting.
Draws 21 hand joints, skeletal bone lines, fingertip indicators, and counts.
"""

from typing import List, Dict, Any
import numpy as np
import cv2

try:
    from hand_detector import HandDetector
except ImportError:
    from .hand_engine import BothHandsEngine


def draw_hands_frame(
    frame: np.ndarray,
    hands: List[Dict[str, Any]],
    summary: Dict[str, Any],
    fps: float = 0.0
) -> np.ndarray:
    """Draws skeletal lines, joint markers, bounding box, and badges on both hands."""
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    # Draw each detected hand
    for hand in hands:
        x1, y1, x2, y2 = hand["bbox"]
        label = hand["handedness"]
        count = hand["finger_count"]
        gesture = hand["gesture"]
        landmarks = hand.get("landmarks", [])

        # Color: Cyan for Left Hand, Neon Green for Right Hand
        color = (255, 200, 0) if "Left" in label else (0, 255, 128)

        # 1. Draw 21 Landmark Bone Connections
        for id1, id2 in HandDetector.HAND_CONNECTIONS:
            if id1 < len(landmarks) and id2 < len(landmarks):
                pt1 = landmarks[id1]
                pt2 = landmarks[id2]
                cv2.line(canvas, pt1, pt2, (40, 40, 50), 3, cv2.LINE_AA)
                cv2.line(canvas, pt1, pt2, color, 1, cv2.LINE_AA)

        # 2. Draw Joint Dots
        for idx, (lx, ly) in enumerate(landmarks):
            # Fingertip indices: 4, 8, 12, 16, 20
            if idx in [4, 8, 12, 16, 20]:
                cv2.circle(canvas, (lx, ly), 6, (0, 230, 255), -1, cv2.LINE_AA)
                cv2.circle(canvas, (lx, ly), 8, (20, 20, 20), 1, cv2.LINE_AA)
            else:
                cv2.circle(canvas, (lx, ly), 3, (255, 255, 255), -1, cv2.LINE_AA)

        # 3. Draw Bounding Box with Corner Accents
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        # 4. Hand Badge above box
        details = hand.get("finger_details", {})
        active_f = [k[0].upper() for k, v in details.items() if v == 1]
        active_str = " ".join(active_f) if active_f else "None"
        badge_text = f"{label}: {count}/5 [{gesture}] ({active_str})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.50
        (tw, th), _ = cv2.getTextSize(badge_text, font, scale, 1)

        bx1 = x1
        if y1 < 60:
            by1 = min(h - 10, y2 + th + 14)
        else:
            by1 = max(th + 12, y1 - 8)

        cv2.rectangle(canvas, (bx1, by1 - th - 6), (bx1 + tw + 12, by1 + 6), (15, 20, 28), -1)
        cv2.rectangle(canvas, (bx1, by1 - th - 6), (bx1 + tw + 12, by1 + 6), color, 2)
        cv2.putText(canvas, badge_text, (bx1 + 6, by1 - 2), font, scale, (255, 255, 255), 1, cv2.LINE_AA)

    # Top info banner
    bar_h = 48
    cv2.rectangle(canvas, (0, 0), (w, bar_h), (12, 16, 23), -1)
    cv2.line(canvas, (0, bar_h), (w, bar_h), (45, 55, 72), 2)

    l_f = summary.get("left_hand_fingers", 0)
    r_f = summary.get("right_hand_fingers", 0)
    tot = summary.get("total_fingers", 0)

    # Left Hand stat
    cv2.putText(canvas, f"Left Hand: {l_f}/5", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 200, 0), 2, cv2.LINE_AA)
    # Right Hand stat
    cv2.putText(canvas, f"Right Hand: {r_f}/5", (220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 128), 2, cv2.LINE_AA)
    # Combined Total stat
    tot_color = (0, 255, 255) if tot > 0 else (180, 180, 180)
    cv2.putText(canvas, f"TOTAL: {tot} / 10 FINGERS", (440, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.68, tot_color, 2, cv2.LINE_AA)
    # FPS
    cv2.putText(canvas, f"{fps:.1f} FPS", (w - 110, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    return canvas
