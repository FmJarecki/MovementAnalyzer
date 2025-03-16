import mediapipe as mp

class PoseDetector:
    def __init__(self, min_detection_confidence: float = 0.9, min_tracking_confidence: float = 0.9):
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(min_detection_confidence=min_detection_confidence,
                                        min_tracking_confidence=min_tracking_confidence)
