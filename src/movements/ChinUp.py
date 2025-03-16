import cv2

from src.calculations import calculate_distance, calculate_angle
from src.movements.PoseDetector import PoseDetector


class ChinUp(PoseDetector):
    BENT_ARM_ANGLE: float = 60.0
    STRAIGHT_ARM_ANGLE: float = 130.0
    BENT_ARM_DISTANCE: float = 0.08
    STRAIGHT_ARM_DISTANCE: float = 0.05
    HIP_DISTANCE: float = 0.15


    def __init__(self):
        PoseDetector.__init__(self)
       
        self._straightened: bool = False

        self._left_arm_angle: float = 0.0
        self._right_arm_angle: float = 0.0
        self._left_arm_distance: float = 0.0
        self._right_arm_distance: float = 0.0
        self._left_distance_arm_change: float = 0.0
        self._right_arm_distance_change: float = 0.0
        self._left_hip_distance: float = 0.0
        self._right_hip_distance: float = 0.0

        self._left_angle_max: float = float('inf')
        self._right_angle_max: float = float('inf')
        self._left_distance_arm_change_min: float = float('inf')
        self._right_arm_distance_change_min: float = float('inf')

    def calc_pull_ups(self, video: str) -> int:
        cap = cv2.VideoCapture(video)

        pull_ups: int = 0
        
        previous_left_distance: float = 0.0
        previous_right_distance: float = 0.0
        previous_left_hip: tuple[float, float] = 0.0, 0.0
        previous_right_hip: tuple[float, float] = 0.0, 0.0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self._pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not results.pose_landmarks:
                continue

            landmarks = results.pose_landmarks.landmark
            left_shoulder: tuple[float, float] = (landmarks[self._mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                                  landmarks[self._mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)
            right_shoulder: tuple[float, float] = (landmarks[self._mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                                   landmarks[self._mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)
            left_elbow: tuple[float, float] = (landmarks[self._mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                               landmarks[self._mp_pose.PoseLandmark.LEFT_ELBOW.value].y)
            right_elbow: tuple[float, float] = (landmarks[self._mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                                landmarks[self._mp_pose.PoseLandmark.RIGHT_ELBOW.value].y)
            left_wrist: tuple[float, float] = (landmarks[self._mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                               landmarks[self._mp_pose.PoseLandmark.LEFT_WRIST.value].y)
            right_wrist: tuple[float, float] = (landmarks[self._mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                                landmarks[self._mp_pose.PoseLandmark.RIGHT_WRIST.value].y)
            left_hip: tuple[float, float] = (landmarks[self._mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                             landmarks[self._mp_pose.PoseLandmark.LEFT_HIP.value].y)
            right_hip: tuple[float, float] = (landmarks[self._mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                              landmarks[self._mp_pose.PoseLandmark.RIGHT_HIP.value].y)

            self._left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            self._right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

            self._left_arm_distance = calculate_distance(left_wrist, left_shoulder)
            self._right_arm_distance = calculate_distance(right_wrist, right_shoulder)

            self._left_distance_arm_change = previous_left_distance - self._left_arm_distance
            self._right_arm_distance_change = previous_right_distance - self._right_arm_distance

            self._left_hip_distance = calculate_distance(left_hip, previous_left_hip)
            self._right_hip_distance = calculate_distance(right_hip, previous_right_hip)

            if self._is_pull_up_detected():
                print("PUll UP")
                pull_ups += 1
                self._straightened = False
                self._reset_max_values()

            if self._is_straightening_detected():
                print("straight")
                self._straightened = True
                previous_left_distance = self._left_arm_distance
                previous_right_distance = self._right_arm_distance
                previous_left_hip = left_hip
                previous_right_hip = right_hip

                self._set_max_values()

            self._mp_drawing.draw_landmarks(frame, results.pose_landmarks, self._mp_pose.POSE_CONNECTIONS)

            cv2.imshow('Video', frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        print(f'Number of pull-ups: {pull_ups}')

        cap.release()
        cv2.destroyAllWindows()

        return pull_ups

    def _is_pull_up_detected(self) -> bool:

        if not self._straightened:
            return False

        are_arms_bent: bool = (self._left_arm_angle < self.BENT_ARM_ANGLE
                               or self._right_arm_angle < self.BENT_ARM_ANGLE)

        are_arms_dislocated: bool = (self._left_distance_arm_change > self.BENT_ARM_DISTANCE
                                    or self._right_arm_distance_change > self.BENT_ARM_DISTANCE)

        are_hips_dislocated: bool = (self._left_hip_distance > self.HIP_DISTANCE
                                    or self._right_hip_distance > self.HIP_DISTANCE)

        pull_up_detected: bool = (are_arms_bent or are_arms_dislocated) and are_hips_dislocated

        return pull_up_detected

    def _is_straightening_detected(self):

        if not self._straightened:
            are_arms_straight: bool = (self._left_arm_angle > self.STRAIGHT_ARM_ANGLE
                                      and self._right_arm_angle > self.STRAIGHT_ARM_ANGLE)

            are_arms_dislocated: bool = (self._left_distance_arm_change < self.STRAIGHT_ARM_DISTANCE
                                        and self._right_arm_distance_change < self.STRAIGHT_ARM_DISTANCE)

            return are_arms_straight and are_arms_dislocated
        else:
            are_arms_straight: bool = (self._left_arm_angle > self._left_angle_max
                                       and self._right_arm_angle > self._right_angle_max)

            are_arms_dislocated: bool = (self._left_distance_arm_change < self._left_distance_arm_change_min
                                         and self._right_arm_distance_change < self._right_arm_distance_change_min)

            return are_arms_straight and are_arms_dislocated

    def _set_max_values(self):
        self._left_angle_max = self._left_arm_angle
        self._right_angle_max = self._right_arm_angle
        self._left_distance_arm_change_min = self._left_distance_arm_change
        self._right_arm_distance_change_min = self._right_arm_distance_change

    def _reset_max_values(self):
        self._left_angle_max = float('inf')
        self._right_angle_max = float('inf')
        self._left_distance_arm_change_min = float('inf')
        self._right_arm_distance_change_min = float('inf')

    def _reset_distances(self):
        self._straightened = False

        self._left_arm_angle = 0.0
        self._right_arm_angle = 0.0
        self._left_arm_distance = 0.0
        self._right_arm_distance = 0.0
        self._left_distance_arm_change = 0.0
        self._right_arm_distance_change = 0.0
        self._left_hip_distance = 0.0
        self._right_hip_distance = 0.0

        self._reset_max_values()
