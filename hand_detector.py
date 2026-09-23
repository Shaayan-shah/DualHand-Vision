"""
Both Hands Tracking & Finger Counting Engine.
Uses MediaPipe HandLandmarker with 21 3D landmarks per hand.
Tracks Left and Right hands simultaneously, counts fingers on each (0 to 5),
computes combined total (0 to 10), and recognizes hand gestures.
"""

import os
import sys
import math
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import mediapipe as mp


class HandDetector:
    """Tracks both hands, counts extended fingers (0-5 per hand, 0-10 total), and classifies gestures."""

    # Hand Landmark Indices (21 total)
    # Wrist: 0
    # Thumb: 1, 2, 3, 4 (Tip: 4, IP: 3, MCP: 2)
    # Index: 5, 6, 7, 8 (Tip: 8, PIP: 6)
    # Middle: 9, 10, 11, 12 (Tip: 12, PIP: 10)
    # Ring: 13, 14, 15, 16 (Tip: 16, PIP: 14)
    # Pinky: 17, 18, 19, 20 (Tip: 20, PIP: 18)

    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # Index
        (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
        (9, 13), (13, 14), (14, 15), (15, 16), # Ring
        (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
        (0, 17)                                # Palm Base
    ]

    def __init__(self, model_path: Optional[str] = None, min_area: int = 2500):
        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            default_path = os.path.join(current_dir, "models", "hand_landmarker.task")
            candidates = [
                default_path,
                os.path.join(os.path.dirname(current_dir), "models", "hand_landmarker.task"),
                "hand_landmarker.task"
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    model_path = cand
                    break

            if model_path is None or not os.path.exists(model_path):
                model_path = default_path
                self._ensure_model_downloaded(model_path)

        self.model_path = model_path
        self.min_area = min_area
        self._init_landmarker()

    @staticmethod
    def _ensure_model_downloaded(target_path: str):
        """Downloads official MediaPipe HandLandmarker task model asset if not present locally."""
        if not os.path.exists(target_path):
            import urllib.request
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            print(f"Retrieving MediaPipe HandLandmarker asset: {target_path}")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, target_path)
            print("MediaPipe model asset verified.")

    def _init_landmarker(self):
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35
        )
        self.landmarker = HandLandmarker.create_from_options(options)

    def process(self, frame: np.ndarray, is_mirrored: bool = False) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Processes frame for both hands and counts extended fingers."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        results = self.landmarker.detect(mp_image)

        hands = []
        if results.hand_landmarks:
            for i, lmarks in enumerate(results.hand_landmarks):
                raw_handedness = results.handedness[i][0].category_name
                conf = float(results.handedness[i][0].score)

                # In mirrored mode (selfie camera), invert handedness to match physical user hands
                if is_mirrored:
                    handedness = "Right" if raw_handedness == "Left" else "Left"
                else:
                    handedness = raw_handedness

                # Convert normalized landmarks to pixel coordinates
                pixel_landmarks = []
                xs, ys = [], []
                for lm in lmarks:
                    px, py = int(lm.x * w), int(lm.y * h)
                    pixel_landmarks.append((px, py))
                    xs.append(px)
                    ys.append(py)

                # Bounding box
                x1 = max(0, min(xs) - 15)
                y1 = max(0, min(ys) - 15)
                x2 = min(w, max(xs) + 15)
                y2 = min(h, max(ys) + 15)

                # Count fingers using rotation-invariant distances
                finger_count, details = self._count_fingers(lmarks, handedness)
                gesture = self._classify_gesture(finger_count, details, lmarks)

                hands.append({
                    "handedness": f"{handedness} Hand",
                    "finger_count": finger_count,
                    "gesture": gesture,
                    "confidence": round(conf, 2),
                    "bbox": [x1, y1, x2, y2],
                    "centroid": (int(np.mean(xs)), int(np.mean(ys))),
                    "landmarks": pixel_landmarks,
                    "finger_details": details
                })

        # Summary computation
        left_fingers = sum(h["finger_count"] for h in hands if "Left" in h["handedness"])
        right_fingers = sum(h["finger_count"] for h in hands if "Right" in h["handedness"])
        total_fingers = left_fingers + right_fingers

        summary = {
            "hands_detected": len(hands),
            "left_hand_fingers": left_fingers,
            "right_hand_fingers": right_fingers,
            "total_fingers": total_fingers,
            "gestures": [f"{h['handedness']}: {h['gesture']}" for h in hands]
        }

        return hands, summary

    def _count_fingers(self, lmarks, handedness: str) -> Tuple[int, Dict[str, int]]:
        """
        Determines if Thumb, Index, Middle, Ring, Pinky are extended.
        Uses rotation-invariant Euclidean distance from wrist joint.
        """
        def dist(p1, p2):
            return math.hypot(p1.x - p2.x, p1.y - p2.y)

        wrist = lmarks[0]

        # Fingers 1 to 4: tip extended further from wrist than PIP and MCP
        index_open = 1 if (dist(lmarks[8], wrist) > dist(lmarks[6], wrist) and dist(lmarks[8], wrist) > dist(lmarks[5], wrist)) else 0
        middle_open = 1 if (dist(lmarks[12], wrist) > dist(lmarks[10], wrist) and dist(lmarks[12], wrist) > dist(lmarks[9], wrist)) else 0
        ring_open = 1 if (dist(lmarks[16], wrist) > dist(lmarks[14], wrist) and dist(lmarks[16], wrist) > dist(lmarks[13], wrist)) else 0
        pinky_open = 1 if (dist(lmarks[20], wrist) > dist(lmarks[18], wrist) and dist(lmarks[20], wrist) > dist(lmarks[17], wrist)) else 0

        # Thumb: checks abduction from pinky MCP (landmark 17)
        thumb_to_pinky = dist(lmarks[4], lmarks[17])
        thumb_ip_to_pinky = dist(lmarks[3], lmarks[17])
        thumb_open = 1 if (thumb_to_pinky > thumb_ip_to_pinky * 1.1) else 0

        details = {
            "thumb": thumb_open,
            "index": index_open,
            "middle": middle_open,
            "ring": ring_open,
            "pinky": pinky_open
        }

        count = sum(details.values())
        return count, details

    def _classify_gesture(self, count: int, details: Dict[str, int], lmarks) -> str:
        """Determines intuitive human gestures based on extended fingers and landmark positions."""
        def dist(p1, p2):
            return math.hypot(p1.x - p2.x, p1.y - p2.y)

        # OK gesture check: thumb and index touching, other three extended
        if details["middle"] == 1 and details["ring"] == 1 and details["pinky"] == 1:
            if dist(lmarks[4], lmarks[8]) < 0.08:
                return "OK Sign"

        # Horns / Rock gesture: index and pinky extended, middle and ring curled
        if details["index"] == 1 and details["pinky"] == 1 and details["middle"] == 0 and details["ring"] == 0:
            return "Rock / Horns"

        if count == 0:
            return "Closed Fist"
        elif count == 1:
            if details["thumb"] == 1:
                return "Thumbs Up"
            return "Pointing / One"
        elif count == 2:
            if details["index"] == 1 and details["middle"] == 1:
                return "Peace / Victory"
            elif details["thumb"] == 1 and details["index"] == 1:
                return "Gun / Two"
            return "Two Fingers"
        elif count == 3:
            return "Three Fingers"
        elif count == 4:
            return "Four Fingers"
        elif count == 5:
            return "Open Palm"

        return f"{count} Fingers"


# Alias for compatibility
BothHandsEngine = HandDetector
